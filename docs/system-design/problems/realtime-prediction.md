# Design a Real-time Prediction System (Amazon Personalize, Netflix)

A production-grade real-time prediction system that delivers personalized predictions with ultra-low latency (<100ms), handling billions of predictions per day with online feature computation, multi-level caching, model ensembling, and intelligent fallback strategies.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M daily active users, 1B predictions/day, 10K QPS sustained, 50K QPS peak, <100ms p99 latency |
| **Key Challenges** | Ultra-low latency inference, online feature computation, feature freshness, multi-level caching, A/B testing, fallback strategies |
| **Core Concepts** | Feature store (online/offline), streaming feature pipeline, model caching, feature caching, dynamic batching, approximate nearest neighbor |
| **Companies** | Amazon Personalize, Netflix, Spotify, TikTok, Pinterest, DoorDash, Uber Eats, YouTube |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Real-time Predictions** | Deliver personalized predictions with <100ms latency | P0 (Must have) |
    | **Online Feature Computation** | Compute features from real-time events (clicks, views) | P0 (Must have) |
    | **Feature Store Integration** | Hybrid online/offline feature store with low latency | P0 (Must have) |
    | **Multi-level Caching** | Cache models, features, and predictions hierarchically | P0 (Must have) |
    | **Model Ensembling** | Combine multiple models for better accuracy | P0 (Must have) |
    | **A/B Testing Framework** | Test model variants with traffic splitting | P0 (Must have) |
    | **Fallback Strategies** | Graceful degradation when services fail | P0 (Must have) |
    | **Streaming Updates** | Update features from real-time event streams | P0 (Must have) |
    | **Batch Precomputation** | Pre-compute predictions for hot users/items | P1 (Should have) |
    | **Embedding Retrieval** | Fast approximate nearest neighbor search | P1 (Should have) |
    | **Model Monitoring** | Track latency, accuracy, feature drift | P1 (Should have) |
    | **Context-aware Predictions** | Factor in time, location, device, session context | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (focus on serving only)
    - Data collection and ETL pipelines
    - User authentication and authorization
    - Billing and payment processing
    - Content creation and management
    - Detailed A/B test analysis and reporting

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (p99)** | < 100ms end-to-end | Critical for user experience in real-time apps |
    | **Latency (p50)** | < 50ms | Most requests should be even faster |
    | **Throughput** | > 50,000 predictions/sec peak | Support high-traffic applications |
    | **Availability** | 99.99% uptime (52 minutes/year) | Revenue-critical system |
    | **Feature Freshness** | < 1 second for real-time features | Capture latest user behavior |
    | **Cache Hit Rate** | > 90% for hot features/models | Reduce backend load |
    | **Accuracy** | > 95% of online model accuracy vs batch | Ensure quality predictions |
    | **Scalability** | Support 10x traffic spikes | Handle viral content, promotions |
    | **Cost Efficiency** | < $0.005 per 1,000 predictions | Competitive pricing at scale |
    | **Fault Tolerance** | Graceful degradation with fallbacks | Never fail completely |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Daily Active Users (DAU): 100M users
    - Average predictions per user: 10 predictions/session
    - Average sessions per user: 3 sessions/day
    - Total predictions: 100M × 10 × 3 = 3B predictions/day (conservative: 1B/day)

    Real-time prediction API:
    - Daily predictions: 1B predictions/day
    - Average QPS: 1B / 86,400 = ~11,600 req/sec
    - Peak QPS: 5x average = ~58,000 req/sec
    - Prediction latency budget: 100ms p99

    Feature computation:
    - Real-time events: 100M users × 50 events/day = 5B events/day
    - Event ingestion QPS: 5B / 86,400 = ~58,000 events/sec
    - Feature extraction: 58,000 events/sec × 20 features = 1.16M feature updates/sec
    - Feature aggregation window: 5 seconds (streaming)

    Model serving:
    - Active models: 50 models (variants for A/B testing)
    - Model versions per A/B test: 2-5 versions
    - Model size: 500 MB - 2 GB per model
    - Model update frequency: 4 times/day (every 6 hours)

    Caching:
    - Cache hit rate target: 90% (L1 + L2 combined)
    - Cache miss penalty: +50ms (feature fetch + model inference)
    - Hot users: 20M users (20% of DAU)
    - Hot items: 1M items (catalog size: 10M items)

    Batch precomputation:
    - Pre-computed predictions: 20M users × 100 items = 2B predictions
    - Batch job frequency: Every 4 hours
    - Batch processing time: 2 hours per job

    Read/Write ratio: 100:1 (inference >> feature updates)
    ```

    ### Storage Estimates

    ```
    Feature store (online):
    - User features: 100M users × 10 KB = 1 TB
    - Item features: 10M items × 5 KB = 50 GB
    - Real-time aggregations: 20M hot users × 20 KB = 400 GB
    - Total online: ~1.5 TB (in-memory Redis cluster)

    Feature store (offline):
    - Historical features: 100M users × 1 MB = 100 TB
    - Feature history (1 year): 5B events/day × 365 × 200 bytes = 365 TB
    - Compressed: 365 TB × 0.3 = ~110 TB
    - Total offline: ~210 TB (S3/HDFS)

    Model storage:
    - Model artifacts: 50 models × 1 GB = 50 GB
    - Model versions (5 versions each): 50 × 5 × 1 GB = 250 GB
    - Model metadata: 50 models × 100 KB = 5 MB
    - Total models: ~250 GB

    Prediction cache:
    - Pre-computed predictions: 2B predictions × 50 bytes = 100 GB
    - Recent predictions (1 hour): 11,600 req/sec × 3,600 × 200 bytes = 8.4 GB
    - Embedding cache: 10M item embeddings × 1 KB = 10 GB
    - Total cache: ~120 GB (Redis)

    Logs and monitoring:
    - Prediction logs: 1B predictions/day × 500 bytes = 500 GB/day
    - 30 days retention: 500 GB × 30 = 15 TB
    - Feature logs: 5B events/day × 200 bytes = 1 TB/day
    - Metrics time-series: 10 KB/sec × 86,400 = 864 MB/day
    - Total logs (30 days): ~45 TB

    Total: 1.5 TB (online) + 210 TB (offline) + 250 GB (models) + 120 GB (cache) + 45 TB (logs) ≈ 260 TB
    ```

    ### Compute Estimates

    ```
    Prediction serving:
    - QPS per instance: 500 req/sec (with caching)
    - Peak QPS: 58,000 req/sec
    - Instances needed: 58,000 / 500 = 116 instances
    - With 2x buffer: 232 instances
    - Instance type: 16 vCPU, 64 GB RAM (c5.4xlarge)
    - Total: 3,712 vCPUs, 14.8 TB RAM

    Model inference (cache misses):
    - Cache miss rate: 10%
    - Inference QPS: 58,000 × 0.1 = 5,800 req/sec
    - Inference time: 20ms per prediction (CPU)
    - GPU instances for deep models: 10 GPUs (T4)
    - CPU instances for tree models: 50 instances (8 vCPU each)

    Feature serving (online store):
    - Feature QPS: 58,000 req/sec (parallel with prediction)
    - Feature fetch time: 1-5ms (Redis)
    - Redis cluster: 20 nodes × 64 GB = 1.28 TB memory
    - Instance type: r5.2xlarge (8 vCPU, 64 GB)

    Streaming feature computation:
    - Event ingestion: 58,000 events/sec
    - Flink/Spark Streaming: 30 instances (8 vCPU each)
    - Kafka cluster: 15 brokers × 8 vCPU = 120 vCPUs
    - Total streaming: 360 vCPUs

    Batch precomputation:
    - Batch jobs: 4 jobs/day
    - Job duration: 2 hours
    - Spark cluster: 200 instances (spot) × 16 vCPU = 3,200 vCPUs
    - Concurrent utilization: 3,200 × (2/24) = 267 vCPU-hours/day

    Control plane:
    - API gateway: 10 instances × 8 vCPU = 80 vCPUs
    - A/B test service: 5 instances × 8 vCPU = 40 vCPUs
    - Model registry: 3 instances × 8 vCPU = 24 vCPUs
    - Monitoring: 5 instances × 16 vCPU = 80 vCPUs

    Total: 3,712 + 400 + 160 + 360 + 224 = 4,856 vCPUs + 10 GPUs (steady state)
    ```

    ### Bandwidth Estimates

    ```
    Prediction API (ingress):
    - 11,600 req/sec × 2 KB (user_id, context, features) = 23 MB/sec ≈ 184 Mbps

    Prediction API (egress):
    - 11,600 req/sec × 5 KB (predictions, scores, metadata) = 58 MB/sec ≈ 464 Mbps

    Event ingestion (ingress):
    - 58,000 events/sec × 500 bytes (event data) = 29 MB/sec ≈ 232 Mbps

    Feature store queries:
    - 11,600 req/sec × 10 KB (features) = 116 MB/sec ≈ 928 Mbps

    Streaming pipeline:
    - Kafka throughput: 58,000 events/sec × 1 KB (enriched) = 58 MB/sec ≈ 464 Mbps
    - Feature updates to online store: 29 MB/sec ≈ 232 Mbps

    Model updates:
    - 50 models × 1 GB × 4 updates/day = 200 GB/day ≈ 19 Mbps average

    Total ingress: ~400 Mbps
    Total egress: ~1.5 Gbps
    ```

    ---

    ## Key Assumptions

    1. 90% of requests can be served from cache (L1 + L2 combined)
    2. Real-time features must be computed within 1 second of event arrival
    3. Model updates happen every 6 hours without service disruption
    4. A/B tests run on 1-20% of traffic with multiple variants
    5. Fallback strategies can maintain 80% prediction quality during failures
    6. Most models are tree-based (XGBoost/LightGBM) with some deep learning
    7. Approximate nearest neighbor can replace exact search with <5% accuracy loss

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Multi-level caching:** L1 (prediction cache) → L2 (feature cache) → L3 (model cache)
    2. **Streaming + Batch:** Real-time feature updates + batch precomputation
    3. **Hybrid feature store:** Low-latency online store + scalable offline store
    4. **Graceful degradation:** Multiple fallback levels for high availability
    5. **A/B testing first:** Built-in experimentation framework
    6. **Approximate algorithms:** Trade exactness for speed (ANN, quantization)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
            API[External API]
        end

        subgraph "Edge & API Layer"
            CDN[CDN/Edge Locations]
            Gateway[API Gateway<br/>Rate Limiting, Auth]
            LB[Load Balancer]
        end

        subgraph "Prediction Service Layer"
            PredService1[Prediction Service 1<br/>Multi-level Cache]
            PredService2[Prediction Service 2<br/>Multi-level Cache]
            PredServiceN[Prediction Service N<br/>Multi-level Cache]
            ABTest[A/B Test Manager<br/>Traffic Routing]
        end

        subgraph "L1: Prediction Cache"
            PredCache[(Redis Cluster<br/>Pre-computed Predictions<br/>100 GB, TTL: 1 hour)]
        end

        subgraph "L2: Feature Cache"
            FeatureCache[(Redis Cluster<br/>User/Item Features<br/>1.5 TB, TTL: 5 min)]
        end

        subgraph "Model Serving Layer"
            ModelServer1[Model Server 1<br/>XGBoost/LightGBM]
            ModelServer2[Model Server 2<br/>Deep Learning]
            ModelServer3[Model Server 3<br/>Embedding ANN]
            ModelCache[(Redis<br/>Model Weights<br/>L3 Cache)]
        end

        subgraph "Feature Store"
            OnlineStore[(Online Store<br/>Redis/DynamoDB<br/>1.5 TB<br/><5ms latency)]
            OfflineStore[(Offline Store<br/>S3/Snowflake<br/>210 TB<br/>Batch features)]
        end

        subgraph "Streaming Feature Pipeline"
            EventStream[Event Stream<br/>Kafka/Kinesis]
            Flink[Flink/Spark Streaming<br/>Real-time Aggregations]
            FeatureProcessor[Feature Processor<br/>5-sec windows]
        end

        subgraph "Batch Pipeline"
            BatchJob[Spark Batch Jobs<br/>Every 6 hours]
            PreCompute[Prediction Precompute<br/>2B predictions]
        end

        subgraph "Model Management"
            ModelRegistry[Model Registry<br/>MLflow/SageMaker]
            ModelStore[(Model Storage<br/>S3/GCS<br/>250 GB)]
            Deployer[Model Deployer<br/>Canary/Blue-Green]
        end

        subgraph "Event Collection"
            EventAPI[Event Collection API]
            EventBuffer[Buffer<br/>Kafka/Kinesis]
        end

        subgraph "Monitoring & Fallback"
            Monitor[Monitoring<br/>Prometheus/Datadog]
            Fallback[Fallback Service<br/>Popular/Trending]
            CircuitBreaker[Circuit Breaker]
        end

        %% Client connections
        Mobile --> CDN
        Web --> CDN
        API --> CDN
        CDN --> Gateway
        Gateway --> LB

        %% Prediction flow
        LB --> PredService1
        LB --> PredService2
        LB --> PredServiceN
        PredService1 --> ABTest
        ABTest --> PredCache

        %% Cache hierarchy
        PredCache -.cache miss.-> FeatureCache
        FeatureCache -.cache miss.-> OnlineStore

        %% Model serving
        PredService1 --> ModelServer1
        PredService1 --> ModelServer2
        PredService1 --> ModelServer3
        ModelServer1 --> ModelCache
        ModelServer2 --> ModelCache
        ModelServer3 --> ModelCache

        %% Feature retrieval
        ModelServer1 --> FeatureCache
        FeatureCache --> OnlineStore

        %% Event ingestion
        Mobile --> EventAPI
        Web --> EventAPI
        EventAPI --> EventBuffer
        EventBuffer --> EventStream

        %% Streaming pipeline
        EventStream --> Flink
        Flink --> FeatureProcessor
        FeatureProcessor --> OnlineStore
        FeatureProcessor --> OfflineStore

        %% Batch pipeline
        OfflineStore --> BatchJob
        BatchJob --> PreCompute
        PreCompute --> PredCache
        BatchJob --> OnlineStore

        %% Model management
        ModelRegistry --> ModelStore
        ModelStore --> Deployer
        Deployer --> ModelCache
        Deployer --> ModelServer1

        %% Monitoring
        PredService1 --> Monitor
        ModelServer1 --> Monitor
        Monitor --> CircuitBreaker
        CircuitBreaker --> Fallback
        Fallback --> PredService1

        style PredCache fill:#e1f5ff
        style FeatureCache fill:#e1f5ff
        style ModelCache fill:#e1f5ff
        style OnlineStore fill:#ffe1f5
        style EventStream fill:#fff4e1
        style Fallback fill:#ffe1e1
    ```

    ---

    ## Component Responsibilities

    ### 1. API Gateway & Load Balancer

    **Responsibilities:**
    - Request routing and load balancing
    - Authentication and rate limiting
    - Request/response transformation
    - TLS termination

    **Technology:**
    - Kong, Envoy, AWS ALB, NGINX

    ---

    ### 2. Prediction Service

    **Responsibilities:**
    - Coordinate prediction workflow
    - Manage multi-level cache hierarchy
    - Handle A/B test traffic routing
    - Orchestrate model serving calls
    - Implement circuit breakers and fallbacks

    **Technology:**
    - Python (FastAPI/Flask), Java (Spring Boot), Go
    - Redis client for caching
    - gRPC for internal communication

    **Key Operations:**
    ```
    GET /predict
    - Check L1 (prediction cache)
    - If miss, check L2 (feature cache) → model inference
    - If miss, fetch from online store → model inference
    - Update caches with TTL
    - Return prediction with metadata
    ```

    ---

    ### 3. Multi-level Cache (L1, L2, L3)

    **L1 - Prediction Cache:**
    - Pre-computed predictions for hot user-item pairs
    - TTL: 1 hour
    - Size: 100 GB (2B predictions × 50 bytes)
    - Hit rate: 50-60%

    **L2 - Feature Cache:**
    - User features, item features, real-time aggregations
    - TTL: 5 minutes (balance freshness vs hit rate)
    - Size: 1.5 TB
    - Hit rate: 30-35%

    **L3 - Model Cache:**
    - Model weights and embeddings
    - TTL: 6 hours (until next model update)
    - Size: 50 GB
    - Hit rate: 99%

    **Technology:**
    - Redis Cluster (distributed, auto-sharding)
    - Memcached (simpler, faster for read-only)

    ---

    ### 4. Model Serving Layer

    **Model Types:**

    | Model Type | Use Case | Latency | Technology |
    |------------|----------|---------|------------|
    | **Tree-based** (XGBoost, LightGBM) | Ranking, CTR prediction | 5-10ms | Treelite, ONNX Runtime |
    | **Deep Learning** (TensorFlow, PyTorch) | Embeddings, neural CF | 20-50ms | TensorFlow Serving, TorchServe |
    | **Embedding ANN** | Similar items, candidate generation | 10-20ms | FAISS, Annoy, ScaNN |
    | **Ensemble** | Combine multiple models | 30-100ms | Custom service |

    **Features:**
    - Dynamic batching (batch requests for GPU efficiency)
    - Model versioning (A/B test variants)
    - Warm model loading (pre-load models into memory)
    - Quantization (INT8 for faster inference)

    ---

    ### 5. Feature Store (Online + Offline)

    **Online Store:**
    - **Purpose:** Low-latency feature serving for real-time inference
    - **Latency:** <5ms p99
    - **Storage:** Redis, DynamoDB, Cassandra
    - **Data:** User profiles, item metadata, real-time aggregations
    - **Update frequency:** Real-time (streaming pipeline)

    **Offline Store:**
    - **Purpose:** Historical features for batch jobs and model training
    - **Latency:** Seconds to minutes
    - **Storage:** S3, Snowflake, BigQuery, Hive
    - **Data:** Feature history, aggregations over long windows
    - **Update frequency:** Batch (hourly/daily)

    **Feature Synchronization:**
    - Streaming pipeline updates both online and offline stores
    - Batch jobs write to offline store, then push hot features to online store
    - Feature consistency via timestamps and versioning

    ---

    ### 6. Streaming Feature Pipeline

    **Workflow:**
    1. **Event Ingestion:** Kafka/Kinesis receives user events (clicks, views, purchases)
    2. **Stream Processing:** Flink/Spark Streaming computes real-time aggregations
    3. **Feature Extraction:** Extract features from event stream (5-second windows)
    4. **Feature Update:** Write to online store (Redis) and offline store (S3)

    **Real-time Features:**
    - Click-through rate (last 1 hour, 1 day)
    - User engagement score (last 5 minutes)
    - Item popularity (last 15 minutes)
    - Session context (current session features)

    **Technology:**
    - Apache Flink (low latency, exactly-once semantics)
    - Kafka Streams (simpler, embedded in app)
    - Spark Streaming (batch-oriented, higher latency)

    ---

    ### 7. Batch Precomputation Pipeline

    **Purpose:**
    - Pre-compute predictions for hot user-item pairs
    - Reduce online inference load
    - Improve cache hit rate

    **Workflow:**
    1. Load user features and item features from offline store
    2. Run batch inference (Spark, 2B predictions in 2 hours)
    3. Store predictions in L1 cache (Redis) with TTL
    4. Update online store with fresh features

    **Schedule:**
    - Every 4-6 hours (balance freshness vs compute cost)
    - Triggered by model updates or feature drift detection

    ---

    ### 8. A/B Testing Framework

    **Responsibilities:**
    - Route traffic to different model variants
    - Track experiment metadata (variant, user assignment)
    - Ensure consistent user experience (sticky assignments)
    - Measure experiment metrics (CTR, engagement, latency)

    **Traffic Routing:**
    - Hash-based routing (user_id % 100 → variant assignment)
    - Weighted routing (90% control, 10% treatment)
    - Multi-armed bandit (dynamic traffic allocation)

    **Technology:**
    - Optimizely, LaunchDarkly (feature flags)
    - Custom service (simpler, lower latency)

    ---

    ### 9. Fallback Service

    **Fallback Hierarchy:**

    ```
    Level 1: Primary prediction service (full features + model)
           ↓ (timeout or error)
    Level 2: Cached predictions (L1 prediction cache)
           ↓ (cache miss)
    Level 3: Simplified model (lightweight model, fewer features)
           ↓ (model unavailable)
    Level 4: Popular items (trending, top-rated, rule-based)
           ↓ (complete failure)
    Level 5: Static defaults (pre-configured recommendations)
    ```

    **Triggers:**
    - Circuit breaker (open after N consecutive failures)
    - Timeout (>100ms SLA breach)
    - Feature store unavailable
    - Model serving error

    ---

    ## Data Flow: End-to-End Prediction

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway
        participant PredService
        participant ABTest
        participant L1Cache as L1: Pred Cache
        participant L2Cache as L2: Feature Cache
        participant ModelServer
        participant OnlineStore
        participant Fallback

        Client->>Gateway: GET /predict?user_id=123
        Gateway->>PredService: Route request
        PredService->>ABTest: Get model variant
        ABTest-->>PredService: variant=model_v2

        PredService->>L1Cache: Get cached prediction
        alt Cache Hit (50%)
            L1Cache-->>PredService: Return prediction
            PredService-->>Client: 200 OK (30ms)
        else Cache Miss
            PredService->>L2Cache: Get user features
            alt Feature Hit (30%)
                L2Cache-->>PredService: Return features
            else Feature Miss
                PredService->>OnlineStore: Fetch features
                OnlineStore-->>PredService: Features (5ms)
                PredService->>L2Cache: Update cache
            end

            PredService->>ModelServer: Inference(features, model_v2)
            alt Success
                ModelServer-->>PredService: Prediction (20ms)
                PredService->>L1Cache: Cache prediction
                PredService-->>Client: 200 OK (80ms)
            else Timeout/Error
                PredService->>Fallback: Get fallback prediction
                Fallback-->>PredService: Popular items
                PredService-->>Client: 200 OK (50ms, degraded)
            end
        end
    ```

    ---

    ## API Design

    ### Prediction API

    ```
    POST /v1/predict

    Request:
    {
      "user_id": "user_12345",
      "context": {
        "device": "mobile",
        "platform": "ios",
        "location": {"lat": 37.7749, "lon": -122.4194},
        "timestamp": 1678901234
      },
      "candidates": ["item_1", "item_2", "item_3"],  // Optional: candidate items
      "num_predictions": 10,
      "model_variant": "auto"  // or specific variant for A/B test
    }

    Response:
    {
      "user_id": "user_12345",
      "predictions": [
        {
          "item_id": "item_789",
          "score": 0.92,
          "rank": 1,
          "explanation": "Based on your recent views"
        },
        {
          "item_id": "item_456",
          "score": 0.87,
          "rank": 2,
          "explanation": "Popular in your area"
        }
      ],
      "metadata": {
        "model_version": "v2.3",
        "experiment_id": "exp_123",
        "latency_ms": 45,
        "cache_hit": false,
        "fallback": false
      }
    }

    Error Response:
    {
      "error": {
        "code": "PREDICTION_TIMEOUT",
        "message": "Prediction service timeout, returning fallback",
        "fallback": true
      },
      "predictions": [...]  // Fallback predictions
    }
    ```

    ### Feature API (Internal)

    ```
    POST /v1/features/batch

    Request:
    {
      "user_ids": ["user_1", "user_2"],
      "item_ids": ["item_1", "item_2"],
      "feature_names": ["user_ctr", "item_popularity", "user_item_affinity"]
    }

    Response:
    {
      "features": {
        "user_1": {
          "user_ctr": 0.05,
          "user_item_affinity": {"item_1": 0.8, "item_2": 0.6}
        },
        "item_1": {
          "item_popularity": 0.92
        }
      },
      "metadata": {
        "latency_ms": 5,
        "cache_hit_rate": 0.85
      }
    }
    ```

=== "🔧 Step 3: Deep Dive"

    ## 3.1 Online Feature Computation with Streaming

    ### Real-time Feature Aggregation

    **Problem:** Compute features from live event stream within 1 second.

    **Solution:** Streaming aggregation with Flink/Kafka Streams.

    ```python
    # Flink streaming job for real-time feature computation
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.functions import KeyedProcessFunction
    from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
    import time
    import json

    class UserFeatureAggregator(KeyedProcessFunction):
        """
        Real-time aggregator for user features with time windows.
        Maintains state for sliding windows (1 min, 5 min, 1 hour).
        """

        def __init__(self):
            self.clicks_1min = None
            self.clicks_5min = None
            self.clicks_1hour = None
            self.last_event_time = None

        def open(self, runtime_context):
            # State for different time windows
            self.clicks_1min = runtime_context.get_state(
                ValueStateDescriptor("clicks_1min", int)
            )
            self.clicks_5min = runtime_context.get_list_state(
                ListStateDescriptor("clicks_5min", tuple)  # (timestamp, count)
            )
            self.clicks_1hour = runtime_context.get_list_state(
                ListStateDescriptor("clicks_1hour", tuple)
            )
            self.last_event_time = runtime_context.get_state(
                ValueStateDescriptor("last_event_time", int)
            )

        def process_element(self, event, ctx):
            """
            Process each user event and update features.
            """
            user_id = event['user_id']
            event_type = event['event_type']  # click, view, purchase
            timestamp = event['timestamp']

            current_time = int(time.time())

            # Update 1-minute window (simple counter)
            if event_type == 'click':
                clicks_1min = self.clicks_1min.value() or 0
                self.clicks_1min.update(clicks_1min + 1)

                # Add to 5-minute window (list of (timestamp, count))
                clicks_5min_list = list(self.clicks_5min.get())
                clicks_5min_list.append((timestamp, 1))

                # Remove old events (>5 minutes)
                clicks_5min_list = [
                    (ts, count) for ts, count in clicks_5min_list
                    if current_time - ts < 300
                ]
                self.clicks_5min.clear()
                self.clicks_5min.add_all(clicks_5min_list)

                # Update 1-hour window similarly
                clicks_1hour_list = list(self.clicks_1hour.get())
                clicks_1hour_list.append((timestamp, 1))
                clicks_1hour_list = [
                    (ts, count) for ts, count in clicks_1hour_list
                    if current_time - ts < 3600
                ]
                self.clicks_1hour.clear()
                self.clicks_1hour.add_all(clicks_1hour_list)

            # Update last event time
            self.last_event_time.update(timestamp)

            # Compute aggregated features
            features = self.compute_features(user_id, current_time)

            # Output features to downstream (Redis/Kafka)
            yield features

        def compute_features(self, user_id, current_time):
            """
            Compute aggregated features from state.
            """
            clicks_1min = self.clicks_1min.value() or 0
            clicks_5min_list = list(self.clicks_5min.get())
            clicks_1hour_list = list(self.clicks_1hour.get())

            clicks_5min = sum(count for _, count in clicks_5min_list)
            clicks_1hour = sum(count for _, count in clicks_1hour_list)

            # Calculate rates
            ctr_5min = clicks_5min / 5.0  # clicks per minute
            ctr_1hour = clicks_1hour / 60.0

            # Recency features
            last_event = self.last_event_time.value() or 0
            seconds_since_last_event = current_time - last_event

            return {
                'user_id': user_id,
                'features': {
                    'clicks_1min': clicks_1min,
                    'clicks_5min': clicks_5min,
                    'clicks_1hour': clicks_1hour,
                    'ctr_5min': ctr_5min,
                    'ctr_1hour': ctr_1hour,
                    'seconds_since_last_event': seconds_since_last_event,
                    'is_active': seconds_since_last_event < 300  # active in last 5 min
                },
                'timestamp': current_time
            }


    # Flink job setup
    def create_streaming_job():
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(30)  # 30 parallel tasks

        # Kafka source
        kafka_source = KafkaSource.builder() \
            .set_bootstrap_servers("kafka:9092") \
            .set_topics("user_events") \
            .set_value_only_deserializer(SimpleStringSchema()) \
            .build()

        # Stream processing pipeline
        stream = env.from_source(kafka_source, WatermarkStrategy.no_watermarks(), "Kafka Source")

        # Parse JSON events
        parsed_stream = stream.map(lambda x: json.loads(x))

        # Key by user_id and aggregate features
        feature_stream = parsed_stream \
            .key_by(lambda event: event['user_id']) \
            .process(UserFeatureAggregator())

        # Sink to Redis (online store)
        feature_stream.add_sink(RedisFeatureSink())

        # Also sink to Kafka for offline storage
        feature_stream.add_sink(KafkaFeatureSink("feature_updates"))

        env.execute("Real-time Feature Computation")
    ```

    ### Feature Store Update (Redis)

    ```python
    import redis
    import json
    from datetime import timedelta

    class OnlineFeatureStore:
        """
        Low-latency online feature store backed by Redis.
        """

        def __init__(self, redis_cluster_endpoints):
            # Redis cluster for horizontal scaling
            from rediscluster import RedisCluster
            self.redis = RedisCluster(
                startup_nodes=redis_cluster_endpoints,
                decode_responses=True,
                max_connections=100
            )

        def update_user_features(self, user_id, features, ttl_seconds=300):
            """
            Update user features with TTL (5 minutes).
            """
            key = f"user_features:{user_id}"

            # Use hash for structured storage
            pipeline = self.redis.pipeline()

            # Set each feature as hash field
            for feature_name, feature_value in features.items():
                pipeline.hset(key, feature_name, json.dumps(feature_value))

            # Set TTL
            pipeline.expire(key, ttl_seconds)

            # Execute atomically
            pipeline.execute()

        def get_user_features(self, user_id, feature_names=None):
            """
            Retrieve user features (used by prediction service).
            """
            key = f"user_features:{user_id}"

            if feature_names:
                # Get specific features
                values = self.redis.hmget(key, feature_names)
                features = {
                    name: json.loads(value) if value else None
                    for name, value in zip(feature_names, values)
                }
            else:
                # Get all features
                features = self.redis.hgetall(key)
                features = {k: json.loads(v) for k, v in features.items()}

            return features

        def batch_get_features(self, user_ids, feature_names):
            """
            Batch fetch features for multiple users (reduce latency).
            """
            pipeline = self.redis.pipeline()

            for user_id in user_ids:
                key = f"user_features:{user_id}"
                pipeline.hmget(key, feature_names)

            results = pipeline.execute()

            # Parse results
            batch_features = {}
            for user_id, values in zip(user_ids, results):
                batch_features[user_id] = {
                    name: json.loads(value) if value else None
                    for name, value in zip(feature_names, values)
                }

            return batch_features
    ```

    ---

    ## 3.2 Multi-level Caching Strategy

    ### Hierarchical Cache Implementation

    ```python
    import hashlib
    import pickle
    from typing import Optional, List, Dict, Any
    import redis
    import time

    class MultiLevelCache:
        """
        Three-level cache hierarchy for prediction system:
        L1: Prediction cache (pre-computed predictions)
        L2: Feature cache (user/item features)
        L3: Model cache (model weights)
        """

        def __init__(self, redis_l1, redis_l2, redis_l3):
            self.l1_pred_cache = redis_l1  # Redis cluster for predictions
            self.l2_feature_cache = redis_l2  # Redis cluster for features
            self.l3_model_cache = redis_l3  # Redis for model weights

            # Cache statistics
            self.stats = {
                'l1_hits': 0, 'l1_misses': 0,
                'l2_hits': 0, 'l2_misses': 0,
                'l3_hits': 0, 'l3_misses': 0
            }

        # === L1: Prediction Cache ===

        def get_prediction(self, user_id: str, context: Dict) -> Optional[Dict]:
            """
            Get pre-computed prediction from L1 cache.
            Cache key includes context for different scenarios.
            """
            cache_key = self._prediction_cache_key(user_id, context)

            start = time.time()
            cached = self.l1_pred_cache.get(cache_key)
            latency = (time.time() - start) * 1000

            if cached:
                self.stats['l1_hits'] += 1
                print(f"L1 HIT: {cache_key} ({latency:.1f}ms)")
                return pickle.loads(cached)
            else:
                self.stats['l1_misses'] += 1
                print(f"L1 MISS: {cache_key} ({latency:.1f}ms)")
                return None

        def set_prediction(self, user_id: str, context: Dict,
                          prediction: Dict, ttl_seconds: int = 3600):
            """
            Store prediction in L1 cache with TTL (1 hour).
            """
            cache_key = self._prediction_cache_key(user_id, context)

            # Serialize prediction
            serialized = pickle.dumps(prediction)

            # Store with expiration
            self.l1_pred_cache.setex(cache_key, ttl_seconds, serialized)

        def _prediction_cache_key(self, user_id: str, context: Dict) -> str:
            """
            Generate cache key that includes context.
            Different contexts (device, location) may have different predictions.
            """
            # Hash context for compact key
            context_str = f"{context.get('device', '')}{context.get('location', '')}"
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            return f"pred:{user_id}:{context_hash}"

        # === L2: Feature Cache ===

        def get_features(self, user_id: str, feature_names: List[str]) -> Optional[Dict]:
            """
            Get features from L2 cache.
            """
            cache_key = f"features:user:{user_id}"

            start = time.time()
            # Get specific feature fields from hash
            values = self.l2_feature_cache.hmget(cache_key, feature_names)
            latency = (time.time() - start) * 1000

            # Check if all features exist
            if all(v is not None for v in values):
                self.stats['l2_hits'] += 1
                print(f"L2 HIT: {cache_key} ({latency:.1f}ms)")

                features = {}
                for name, value in zip(feature_names, values):
                    features[name] = pickle.loads(value)
                return features
            else:
                self.stats['l2_misses'] += 1
                print(f"L2 MISS: {cache_key} ({latency:.1f}ms)")
                return None

        def set_features(self, user_id: str, features: Dict, ttl_seconds: int = 300):
            """
            Store features in L2 cache with TTL (5 minutes).
            """
            cache_key = f"features:user:{user_id}"

            # Use Redis hash for structured storage
            pipeline = self.l2_feature_cache.pipeline()

            for feature_name, feature_value in features.items():
                serialized = pickle.dumps(feature_value)
                pipeline.hset(cache_key, feature_name, serialized)

            pipeline.expire(cache_key, ttl_seconds)
            pipeline.execute()

        # === L3: Model Cache ===

        def get_model(self, model_id: str, version: str) -> Optional[Any]:
            """
            Get model from L3 cache.
            Models are large, so cache hit is critical.
            """
            cache_key = f"model:{model_id}:{version}"

            start = time.time()
            cached = self.l3_model_cache.get(cache_key)
            latency = (time.time() - start) * 1000

            if cached:
                self.stats['l3_hits'] += 1
                print(f"L3 HIT: {cache_key} ({latency:.1f}ms)")
                return pickle.loads(cached)
            else:
                self.stats['l3_misses'] += 1
                print(f"L3 MISS: {cache_key} ({latency:.1f}ms)")
                return None

        def set_model(self, model_id: str, version: str,
                     model: Any, ttl_seconds: int = 21600):
            """
            Store model in L3 cache with TTL (6 hours).
            """
            cache_key = f"model:{model_id}:{version}"

            # Serialize model
            serialized = pickle.dumps(model)

            # Store with long TTL (models change infrequently)
            self.l3_model_cache.setex(cache_key, ttl_seconds, serialized)

        # === Cache Statistics ===

        def get_stats(self) -> Dict:
            """
            Get cache hit rates and statistics.
            """
            total_l1 = self.stats['l1_hits'] + self.stats['l1_misses']
            total_l2 = self.stats['l2_hits'] + self.stats['l2_misses']
            total_l3 = self.stats['l3_hits'] + self.stats['l3_misses']

            return {
                'l1_hit_rate': self.stats['l1_hits'] / total_l1 if total_l1 > 0 else 0,
                'l2_hit_rate': self.stats['l2_hits'] / total_l2 if total_l2 > 0 else 0,
                'l3_hit_rate': self.stats['l3_hits'] / total_l3 if total_l3 > 0 else 0,
                'combined_hit_rate': (self.stats['l1_hits'] + self.stats['l2_hits']) /
                                    (total_l1 + total_l2) if (total_l1 + total_l2) > 0 else 0,
                **self.stats
            }
    ```

    ---

    ## 3.3 Model Serving with Ensembling

    ### Multi-model Ensemble Serving

    ```python
    from typing import List, Dict, Any
    import numpy as np
    import xgboost as xgb
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    class EnsemblePredictionService:
        """
        Serve multiple models and combine predictions.
        Models: XGBoost (ranking), Deep Learning (embeddings), ANN (candidates).
        """

        def __init__(self, cache: MultiLevelCache, feature_store: OnlineFeatureStore):
            self.cache = cache
            self.feature_store = feature_store
            self.executor = ThreadPoolExecutor(max_workers=10)

            # Model configurations
            self.models = {
                'xgboost_ranker': {
                    'weight': 0.5,
                    'type': 'xgboost',
                    'model_id': 'ranker_v1'
                },
                'neural_cf': {
                    'weight': 0.3,
                    'type': 'deep_learning',
                    'model_id': 'ncf_v2'
                },
                'item_ann': {
                    'weight': 0.2,
                    'type': 'embedding_ann',
                    'model_id': 'ann_v1'
                }
            }

        def predict(self, user_id: str, context: Dict,
                   candidates: List[str] = None, num_results: int = 10) -> Dict:
            """
            Main prediction endpoint with multi-level caching.
            """
            start_time = time.time()

            # === L1: Check prediction cache ===
            cached_pred = self.cache.get_prediction(user_id, context)
            if cached_pred:
                cached_pred['latency_ms'] = (time.time() - start_time) * 1000
                cached_pred['cache_level'] = 'L1'
                return cached_pred

            # === L2: Check feature cache + model inference ===
            features = self._get_features_with_cache(user_id, context)

            if features is None:
                # Fallback if features unavailable
                return self._fallback_prediction(user_id, context)

            # === Model Inference (Parallel Ensemble) ===
            predictions = self._ensemble_predict(user_id, features, candidates, context)

            # Cache prediction for future requests
            self.cache.set_prediction(user_id, context, predictions, ttl_seconds=3600)

            predictions['latency_ms'] = (time.time() - start_time) * 1000
            predictions['cache_level'] = 'none'

            return predictions

        def _get_features_with_cache(self, user_id: str, context: Dict) -> Optional[Dict]:
            """
            Get features with L2 cache, fallback to feature store.
            """
            required_features = [
                'user_ctr_1hour', 'user_ctr_5min', 'user_embedding',
                'user_recent_items', 'user_category_prefs'
            ]

            # Try L2 cache
            features = self.cache.get_features(user_id, required_features)

            if features is None:
                # Cache miss: fetch from feature store
                features = self.feature_store.get_user_features(user_id, required_features)

                if features:
                    # Update L2 cache
                    self.cache.set_features(user_id, features, ttl_seconds=300)
                else:
                    # Feature store miss
                    return None

            # Enrich with context features
            features['device'] = context.get('device', 'unknown')
            features['time_of_day'] = self._get_time_of_day(context.get('timestamp'))

            return features

        def _ensemble_predict(self, user_id: str, features: Dict,
                             candidates: List[str], context: Dict) -> Dict:
            """
            Run multiple models in parallel and ensemble predictions.
            """
            # Submit model inference tasks in parallel
            futures = {}

            for model_name, config in self.models.items():
                future = self.executor.submit(
                    self._run_model, model_name, config, user_id, features, candidates, context
                )
                futures[future] = model_name

            # Collect results as they complete
            model_predictions = {}
            for future in as_completed(futures, timeout=0.08):  # 80ms timeout
                model_name = futures[future]
                try:
                    result = future.result()
                    model_predictions[model_name] = result
                except Exception as e:
                    print(f"Model {model_name} failed: {e}")
                    # Continue with other models

            # Ensemble predictions
            final_predictions = self._combine_predictions(model_predictions)

            return {
                'user_id': user_id,
                'predictions': final_predictions[:10],  # Top 10
                'model_versions': {k: self.models[k]['model_id'] for k in model_predictions.keys()},
                'ensemble_weights': {k: self.models[k]['weight'] for k in self.models.keys()}
            }

        def _run_model(self, model_name: str, config: Dict,
                      user_id: str, features: Dict,
                      candidates: List[str], context: Dict) -> List[Dict]:
            """
            Run a single model inference.
            """
            model_type = config['type']
            model_id = config['model_id']
            version = 'v1'  # Get from A/B test service

            # === L3: Check model cache ===
            model = self.cache.get_model(model_id, version)

            if model is None:
                # Load model from model store
                model = self._load_model(model_id, version)

                if model:
                    # Cache model
                    self.cache.set_model(model_id, version, model, ttl_seconds=21600)
                else:
                    raise Exception(f"Model {model_id} not found")

            # Run inference based on model type
            if model_type == 'xgboost':
                return self._xgboost_inference(model, features, candidates)
            elif model_type == 'deep_learning':
                return self._deep_learning_inference(model, features, candidates)
            elif model_type == 'embedding_ann':
                return self._ann_inference(model, features, candidates)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        def _xgboost_inference(self, model: xgb.Booster,
                              features: Dict, candidates: List[str]) -> List[Dict]:
            """
            XGBoost inference for ranking candidates.
            """
            # Prepare feature matrix
            feature_matrix = self._prepare_xgboost_features(features, candidates)

            # Predict scores
            dmatrix = xgb.DMatrix(feature_matrix)
            scores = model.predict(dmatrix)

            # Return ranked candidates
            results = []
            for candidate_id, score in zip(candidates, scores):
                results.append({
                    'item_id': candidate_id,
                    'score': float(score),
                    'model': 'xgboost_ranker'
                })

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results

        def _deep_learning_inference(self, model: torch.nn.Module,
                                    features: Dict, candidates: List[str]) -> List[Dict]:
            """
            Deep learning inference (e.g., Neural Collaborative Filtering).
            """
            model.eval()

            with torch.no_grad():
                # User embedding
                user_embedding = torch.tensor(features['user_embedding']).float()

                # Item embeddings (batch)
                item_embeddings = torch.stack([
                    torch.tensor(self._get_item_embedding(item_id)).float()
                    for item_id in candidates
                ])

                # Compute similarity scores
                scores = torch.matmul(item_embeddings, user_embedding)

                # Return results
                results = []
                for candidate_id, score in zip(candidates, scores):
                    results.append({
                        'item_id': candidate_id,
                        'score': float(score),
                        'model': 'neural_cf'
                    })

                results.sort(key=lambda x: x['score'], reverse=True)
                return results

        def _ann_inference(self, index, features: Dict,
                          candidates: List[str] = None) -> List[Dict]:
            """
            Approximate Nearest Neighbor for candidate retrieval.
            Uses FAISS for fast similarity search.
            """
            import faiss

            # User embedding
            user_embedding = np.array(features['user_embedding']).astype('float32')
            user_embedding = user_embedding.reshape(1, -1)

            # Search top-k similar items
            k = 100 if candidates is None else len(candidates)
            distances, indices = index.search(user_embedding, k)

            # Return results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                item_id = self._get_item_id_from_index(idx)
                results.append({
                    'item_id': item_id,
                    'score': float(1.0 / (1.0 + distance)),  # Convert distance to score
                    'model': 'item_ann'
                })

            return results

        def _combine_predictions(self, model_predictions: Dict[str, List[Dict]]) -> List[Dict]:
            """
            Ensemble predictions using weighted average of scores.
            """
            # Aggregate scores per item
            item_scores = {}

            for model_name, predictions in model_predictions.items():
                weight = self.models[model_name]['weight']

                for pred in predictions:
                    item_id = pred['item_id']
                    score = pred['score']

                    if item_id not in item_scores:
                        item_scores[item_id] = 0.0

                    item_scores[item_id] += weight * score

            # Convert to list and sort
            combined = [
                {'item_id': item_id, 'score': score}
                for item_id, score in item_scores.items()
            ]
            combined.sort(key=lambda x: x['score'], reverse=True)

            return combined

        def _fallback_prediction(self, user_id: str, context: Dict) -> Dict:
            """
            Fallback when features or models unavailable.
            Return popular/trending items.
            """
            # Simple fallback: popular items
            popular_items = self._get_popular_items(context.get('category'))

            return {
                'user_id': user_id,
                'predictions': popular_items[:10],
                'fallback': True,
                'fallback_reason': 'features_unavailable'
            }

        def _get_popular_items(self, category: str = None) -> List[Dict]:
            """
            Get popular/trending items as fallback.
            """
            # This would query a trending items cache
            # For now, return dummy data
            return [
                {'item_id': f'item_{i}', 'score': 1.0 - i * 0.05, 'reason': 'popular'}
                for i in range(20)
            ]

        # Helper methods

        def _load_model(self, model_id: str, version: str):
            """Load model from model registry/storage."""
            # Implementation depends on model storage (S3, MLflow, etc.)
            pass

        def _prepare_xgboost_features(self, features: Dict, candidates: List[str]):
            """Prepare feature matrix for XGBoost."""
            # Implementation depends on feature schema
            pass

        def _get_item_embedding(self, item_id: str):
            """Get item embedding from feature store."""
            # Implementation depends on feature store
            pass

        def _get_item_id_from_index(self, idx: int) -> str:
            """Map FAISS index to item ID."""
            # Implementation depends on index structure
            pass

        def _get_time_of_day(self, timestamp: int) -> str:
            """Extract time of day feature."""
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour

            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
    ```

    ---

    ## 3.4 Fallback Strategies with Circuit Breaker

    ### Intelligent Fallback System

    ```python
    from enum import Enum
    from typing import Optional, Dict, Callable
    import time
    from collections import deque

    class FallbackLevel(Enum):
        """Fallback hierarchy levels."""
        PRIMARY = 1          # Full prediction with all models
        CACHED = 2           # Cached predictions
        SIMPLIFIED = 3       # Lightweight model
        POPULAR = 4          # Popular/trending items
        STATIC = 5           # Pre-configured defaults

    class CircuitBreaker:
        """
        Circuit breaker to prevent cascading failures.
        States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery).
        """

        def __init__(self, failure_threshold: int = 5,
                    recovery_timeout: int = 60,
                    success_threshold: int = 2):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.success_threshold = success_threshold

            # State
            self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

            # Metrics
            self.recent_calls = deque(maxlen=100)  # Track recent call results

        def call(self, func: Callable, *args, **kwargs):
            """
            Execute function with circuit breaker protection.
            """
            # Check if circuit is open
            if self.state == 'OPEN':
                # Check if recovery timeout elapsed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    print("Circuit breaker: OPEN -> HALF_OPEN (testing recovery)")
                    self.state = 'HALF_OPEN'
                    self.success_count = 0
                else:
                    # Circuit open, raise exception immediately
                    raise Exception("Circuit breaker OPEN")

            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time

                # Success
                self._record_success(latency)
                return result

            except Exception as e:
                latency = time.time() - start_time

                # Failure
                self._record_failure(latency, e)
                raise

        def _record_success(self, latency: float):
            """Record successful call."""
            self.recent_calls.append({'success': True, 'latency': latency, 'time': time.time()})

            if self.state == 'HALF_OPEN':
                self.success_count += 1

                # Check if enough successes to close circuit
                if self.success_count >= self.success_threshold:
                    print("Circuit breaker: HALF_OPEN -> CLOSED (recovered)")
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    self.success_count = 0

            elif self.state == 'CLOSED':
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count -= 1

        def _record_failure(self, latency: float, exception: Exception):
            """Record failed call."""
            self.recent_calls.append({
                'success': False,
                'latency': latency,
                'time': time.time(),
                'error': str(exception)
            })

            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == 'HALF_OPEN':
                # Failure during recovery, re-open circuit
                print("Circuit breaker: HALF_OPEN -> OPEN (recovery failed)")
                self.state = 'OPEN'
                self.success_count = 0

            elif self.state == 'CLOSED':
                # Check if threshold exceeded
                if self.failure_count >= self.failure_threshold:
                    print(f"Circuit breaker: CLOSED -> OPEN ({self.failure_count} failures)")
                    self.state = 'OPEN'

        def get_stats(self) -> Dict:
            """Get circuit breaker statistics."""
            total_calls = len(self.recent_calls)
            if total_calls == 0:
                return {'state': self.state, 'calls': 0}

            successes = sum(1 for call in self.recent_calls if call['success'])
            failures = total_calls - successes
            avg_latency = sum(call['latency'] for call in self.recent_calls) / total_calls

            return {
                'state': self.state,
                'total_calls': total_calls,
                'successes': successes,
                'failures': failures,
                'success_rate': successes / total_calls,
                'avg_latency': avg_latency,
                'failure_count': self.failure_count
            }


    class FallbackPredictionService:
        """
        Prediction service with multi-level fallback strategy.
        """

        def __init__(self, ensemble_service: EnsemblePredictionService):
            self.ensemble_service = ensemble_service

            # Circuit breakers for each service
            self.circuit_breakers = {
                'prediction_service': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
                'feature_store': CircuitBreaker(failure_threshold=10, recovery_timeout=30),
                'model_serving': CircuitBreaker(failure_threshold=5, recovery_timeout=60)
            }

            # Fallback cache
            self.popular_items_cache = {}
            self.static_defaults = self._load_static_defaults()

        def predict_with_fallback(self, user_id: str, context: Dict,
                                 num_results: int = 10) -> Dict:
            """
            Predict with automatic fallback on failures.
            """
            start_time = time.time()
            fallback_level = FallbackLevel.PRIMARY

            try:
                # Level 1: Primary prediction service
                result = self.circuit_breakers['prediction_service'].call(
                    self.ensemble_service.predict,
                    user_id, context, num_results=num_results
                )
                return self._add_metadata(result, fallback_level, start_time)

            except Exception as e:
                print(f"Primary prediction failed: {e}")
                fallback_level = FallbackLevel.CACHED

            try:
                # Level 2: Cached predictions
                result = self._get_cached_prediction(user_id, context)
                if result:
                    return self._add_metadata(result, fallback_level, start_time)

                fallback_level = FallbackLevel.SIMPLIFIED

            except Exception as e:
                print(f"Cached prediction failed: {e}")

            try:
                # Level 3: Simplified model (fewer features, faster)
                result = self._simplified_prediction(user_id, context, num_results)
                return self._add_metadata(result, fallback_level, start_time)

            except Exception as e:
                print(f"Simplified prediction failed: {e}")
                fallback_level = FallbackLevel.POPULAR

            try:
                # Level 4: Popular items
                result = self._popular_items_prediction(user_id, context, num_results)
                return self._add_metadata(result, fallback_level, start_time)

            except Exception as e:
                print(f"Popular items failed: {e}")
                fallback_level = FallbackLevel.STATIC

            # Level 5: Static defaults (always succeeds)
            result = self._static_defaults_prediction(user_id, context, num_results)
            return self._add_metadata(result, fallback_level, start_time)

        def _get_cached_prediction(self, user_id: str, context: Dict) -> Optional[Dict]:
            """Get prediction from L1 cache."""
            return self.ensemble_service.cache.get_prediction(user_id, context)

        def _simplified_prediction(self, user_id: str, context: Dict,
                                  num_results: int) -> Dict:
            """
            Simplified prediction with lightweight model and minimal features.
            """
            # Use only cached features (no feature store lookup)
            features = self.ensemble_service.cache.get_features(
                user_id, ['user_ctr_1hour', 'user_recent_items']
            )

            if not features:
                raise Exception("No cached features available")

            # Use simple model (e.g., popularity-based with personalization)
            predictions = self._simple_personalized_ranking(features, num_results)

            return {
                'user_id': user_id,
                'predictions': predictions,
                'model': 'simplified'
            }

        def _popular_items_prediction(self, user_id: str, context: Dict,
                                     num_results: int) -> Dict:
            """
            Return popular/trending items.
            """
            category = context.get('category', 'default')

            # Check cache
            if category in self.popular_items_cache:
                popular = self.popular_items_cache[category]
            else:
                # Fetch from database (cached separately)
                popular = self._fetch_popular_items(category, num_results * 2)
                self.popular_items_cache[category] = popular

            return {
                'user_id': user_id,
                'predictions': popular[:num_results],
                'model': 'popular'
            }

        def _static_defaults_prediction(self, user_id: str, context: Dict,
                                       num_results: int) -> Dict:
            """
            Return pre-configured static defaults.
            """
            return {
                'user_id': user_id,
                'predictions': self.static_defaults[:num_results],
                'model': 'static_default'
            }

        def _add_metadata(self, result: Dict, fallback_level: FallbackLevel,
                         start_time: float) -> Dict:
            """Add metadata to result."""
            result['fallback_level'] = fallback_level.name
            result['fallback_order'] = fallback_level.value
            result['total_latency_ms'] = (time.time() - start_time) * 1000
            return result

        def _simple_personalized_ranking(self, features: Dict, num_results: int) -> List[Dict]:
            """Simple personalized ranking using basic features."""
            # Implementation: mix popular items with user preferences
            pass

        def _fetch_popular_items(self, category: str, limit: int) -> List[Dict]:
            """Fetch popular items from database."""
            # Implementation: query trending items
            pass

        def _load_static_defaults(self) -> List[Dict]:
            """Load pre-configured default recommendations."""
            # Implementation: load from config file
            return [
                {'item_id': f'default_item_{i}', 'score': 1.0 - i * 0.05}
                for i in range(50)
            ]

        def get_health_status(self) -> Dict:
            """Get health status of all services."""
            return {
                service: cb.get_stats()
                for service, cb in self.circuit_breakers.items()
            }
    ```

    ---

    ## 3.5 A/B Testing with Traffic Routing

    ### A/B Test Framework

    ```python
    import hashlib
    from typing import Dict, Optional
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class Experiment:
        """A/B test experiment configuration."""
        experiment_id: str
        name: str
        variants: Dict[str, Dict]  # variant_name -> config
        traffic_allocation: Dict[str, float]  # variant_name -> percentage
        start_time: datetime
        end_time: datetime
        enabled: bool
        sticky: bool  # Keep users in same variant

    class ABTestManager:
        """
        A/B testing framework for model experimentation.
        """

        def __init__(self, config_store):
            self.config_store = config_store
            self.active_experiments = {}
            self.user_assignments = {}  # Cache for sticky assignments
            self._load_experiments()

        def _load_experiments(self):
            """Load active experiments from config store."""
            experiments = self.config_store.get_active_experiments()

            for exp_config in experiments:
                experiment = Experiment(**exp_config)
                self.active_experiments[experiment.experiment_id] = experiment

        def get_variant(self, user_id: str, experiment_id: str,
                       context: Dict = None) -> Optional[str]:
            """
            Assign user to experiment variant.
            Uses consistent hashing for deterministic assignment.
            """
            experiment = self.active_experiments.get(experiment_id)

            if not experiment or not experiment.enabled:
                return None

            # Check if experiment is active
            now = datetime.now()
            if not (experiment.start_time <= now <= experiment.end_time):
                return None

            # Check sticky assignment
            if experiment.sticky:
                cache_key = f"{user_id}:{experiment_id}"
                if cache_key in self.user_assignments:
                    return self.user_assignments[cache_key]

            # Assign variant using consistent hashing
            variant = self._assign_variant(user_id, experiment)

            # Cache assignment if sticky
            if experiment.sticky:
                self.user_assignments[cache_key] = variant

            return variant

        def _assign_variant(self, user_id: str, experiment: Experiment) -> str:
            """
            Assign variant using consistent hashing.
            Ensures stable assignment and correct traffic split.
            """
            # Hash user_id + experiment_id for deterministic assignment
            hash_input = f"{user_id}:{experiment.experiment_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

            # Normalize to [0, 1]
            normalized_hash = (hash_value % 100000) / 100000.0

            # Assign variant based on traffic allocation
            cumulative = 0.0
            for variant_name, allocation in experiment.traffic_allocation.items():
                cumulative += allocation
                if normalized_hash <= cumulative:
                    return variant_name

            # Default to control
            return 'control'

        def get_variant_config(self, user_id: str, experiment_id: str,
                              context: Dict = None) -> Dict:
            """
            Get variant configuration for user.
            """
            variant = self.get_variant(user_id, experiment_id, context)

            if not variant:
                # No experiment, return default config
                return {'model_version': 'production', 'features': 'full'}

            experiment = self.active_experiments[experiment_id]
            variant_config = experiment.variants.get(variant, {})

            return {
                'experiment_id': experiment_id,
                'variant': variant,
                **variant_config
            }

        def track_event(self, user_id: str, experiment_id: str,
                       event_type: str, event_data: Dict):
            """
            Track experiment event (exposure, click, conversion).
            """
            variant = self.get_variant(user_id, experiment_id)

            if not variant:
                return

            # Send event to analytics pipeline
            event = {
                'user_id': user_id,
                'experiment_id': experiment_id,
                'variant': variant,
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': datetime.now().isoformat()
            }

            # Write to Kafka or analytics service
            self._send_to_analytics(event)

        def _send_to_analytics(self, event: Dict):
            """Send event to analytics pipeline."""
            # Implementation: send to Kafka, Kinesis, or analytics API
            pass


    class PredictionServiceWithABTest:
        """
        Prediction service integrated with A/B testing.
        """

        def __init__(self, fallback_service: FallbackPredictionService,
                    ab_test_manager: ABTestManager):
            self.fallback_service = fallback_service
            self.ab_test_manager = ab_test_manager

        def predict(self, user_id: str, context: Dict,
                   num_results: int = 10) -> Dict:
            """
            Predict with A/B test variant assignment.
            """
            # Get experiment variant
            experiment_id = 'model_serving_v2_vs_v3'
            variant_config = self.ab_test_manager.get_variant_config(
                user_id, experiment_id, context
            )

            # Track exposure
            self.ab_test_manager.track_event(
                user_id, experiment_id, 'exposure', {'context': context}
            )

            # Override model version based on variant
            if variant_config.get('variant') == 'treatment':
                context['model_version'] = variant_config.get('model_version', 'v3')
            else:
                context['model_version'] = 'v2'  # control

            # Run prediction
            result = self.fallback_service.predict_with_fallback(
                user_id, context, num_results
            )

            # Add experiment metadata
            result['experiment'] = {
                'experiment_id': experiment_id,
                'variant': variant_config.get('variant', 'none')
            }

            return result
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## 4.1 Feature Precomputation & Materialization

    **Problem:** Computing features on-the-fly adds latency (5-20ms).

    **Solution:** Pre-compute and materialize hot features.

    ### Batch Feature Computation

    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, window, count, avg, collect_list
    from pyspark.sql.types import StringType, FloatType, ArrayType
    import time

    class BatchFeatureComputation:
        """
        Batch compute features and push to online store.
        Runs every 4-6 hours.
        """

        def __init__(self, spark: SparkSession, online_store: OnlineFeatureStore):
            self.spark = spark
            self.online_store = online_store

        def compute_user_features(self, events_path: str, users_path: str):
            """
            Compute user features from historical events.
            """
            # Load events (Parquet/Avro from S3/HDFS)
            events = self.spark.read.parquet(events_path)

            # Filter last 7 days
            cutoff_time = int(time.time()) - 7 * 86400
            recent_events = events.filter(col('timestamp') > cutoff_time)

            # Compute aggregations
            user_features = recent_events.groupBy('user_id').agg(
                count('*').alias('total_events'),
                count(col('event_type') == 'click').alias('total_clicks'),
                count(col('event_type') == 'purchase').alias('total_purchases'),
                avg('session_duration').alias('avg_session_duration'),
                collect_list('item_id').alias('recent_items')  # Last N items
            )

            # Compute derived features
            user_features = user_features.withColumn(
                'ctr',
                col('total_clicks') / col('total_events')
            ).withColumn(
                'purchase_rate',
                col('total_purchases') / col('total_clicks')
            )

            # Write to online store (parallel)
            user_features.foreachPartition(self._write_partition_to_online_store)

            print(f"Computed features for {user_features.count()} users")

        def _write_partition_to_online_store(self, partition):
            """
            Write partition of features to online store.
            Uses batch writes for efficiency.
            """
            batch = []

            for row in partition:
                features = {
                    'total_events': row['total_events'],
                    'total_clicks': row['total_clicks'],
                    'ctr': row['ctr'],
                    'purchase_rate': row['purchase_rate'],
                    'recent_items': row['recent_items'][:50]  # Limit size
                }

                batch.append((row['user_id'], features))

                # Flush batch every 100 users
                if len(batch) >= 100:
                    self._flush_batch(batch)
                    batch = []

            # Flush remaining
            if batch:
                self._flush_batch(batch)

        def _flush_batch(self, batch):
            """Flush batch of features to Redis."""
            pipeline = self.online_store.redis.pipeline()

            for user_id, features in batch:
                self.online_store.update_user_features(
                    user_id, features, ttl_seconds=21600  # 6 hours
                )

            pipeline.execute()
    ```

    ---

    ## 4.2 Approximate Nearest Neighbor (ANN) for Embeddings

    **Problem:** Exact nearest neighbor search is slow (O(n) for millions of items).

    **Solution:** Use approximate algorithms (FAISS, Annoy) for 10-100x speedup.

    ### FAISS-based Embedding Search

    ```python
    import faiss
    import numpy as np
    from typing import List, Tuple

    class EmbeddingSearchIndex:
        """
        Fast embedding search using FAISS.
        """

        def __init__(self, embedding_dim: int = 256):
            self.embedding_dim = embedding_dim
            self.index = None
            self.item_ids = []  # Map index position to item_id

        def build_index(self, embeddings: np.ndarray, item_ids: List[str],
                       index_type: str = 'IVF'):
            """
            Build FAISS index from embeddings.

            Index types:
            - Flat: Exact search, O(n) but accurate
            - IVF: Inverted file, faster but approximate
            - HNSW: Hierarchical navigable small world, very fast
            """
            n_embeddings = embeddings.shape[0]

            if index_type == 'Flat':
                # Exact search (baseline)
                self.index = faiss.IndexFlatL2(self.embedding_dim)

            elif index_type == 'IVF':
                # Inverted file index (faster, approximate)
                n_clusters = min(100, n_embeddings // 100)  # 100 items per cluster
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)

                # Train index
                self.index.train(embeddings)

                # Set search parameters
                self.index.nprobe = 10  # Search 10 clusters (balance speed/accuracy)

            elif index_type == 'HNSW':
                # Hierarchical NSW (fastest, good accuracy)
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 = M parameter

            else:
                raise ValueError(f"Unknown index type: {index_type}")

            # Add embeddings to index
            self.index.add(embeddings)
            self.item_ids = item_ids

            print(f"Built {index_type} index with {n_embeddings} embeddings")

        def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
            """
            Search for k nearest neighbors.

            Returns:
                List of (item_id, distance) tuples.
            """
            # Reshape for FAISS (expects 2D array)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Search
            distances, indices = self.index.search(query_embedding, k)

            # Map indices to item IDs
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.item_ids):  # Valid index
                    item_id = self.item_ids[idx]
                    # Convert L2 distance to similarity score
                    score = 1.0 / (1.0 + distance)
                    results.append((item_id, score))

            return results

        def save_index(self, filepath: str):
            """Save index to disk."""
            faiss.write_index(self.index, filepath)

        def load_index(self, filepath: str):
            """Load index from disk."""
            self.index = faiss.read_index(filepath)

        def benchmark(self, query_embeddings: np.ndarray, k: int = 10):
            """
            Benchmark search performance.
            """
            import time

            n_queries = query_embeddings.shape[0]

            start = time.time()
            for i in range(n_queries):
                self.search(query_embeddings[i], k)
            elapsed = time.time() - start

            qps = n_queries / elapsed
            latency_ms = (elapsed / n_queries) * 1000

            print(f"Benchmark results:")
            print(f"  Queries: {n_queries}")
            print(f"  QPS: {qps:.1f}")
            print(f"  Avg latency: {latency_ms:.2f}ms")

            return {'qps': qps, 'latency_ms': latency_ms}


    # Example usage
    def build_item_embedding_index():
        """
        Build and test embedding index.
        """
        # Load item embeddings (from model or feature store)
        item_ids = [f'item_{i}' for i in range(1_000_000)]  # 1M items
        embeddings = np.random.rand(1_000_000, 256).astype('float32')  # 256-dim

        # Build indexes with different types
        for index_type in ['Flat', 'IVF', 'HNSW']:
            print(f"\n=== {index_type} Index ===")

            index = EmbeddingSearchIndex(embedding_dim=256)
            index.build_index(embeddings, item_ids, index_type)

            # Benchmark
            test_queries = np.random.rand(1000, 256).astype('float32')
            index.benchmark(test_queries, k=10)

            # Test accuracy (compare with Flat)
            if index_type != 'Flat':
                flat_index = EmbeddingSearchIndex(embedding_dim=256)
                flat_index.build_index(embeddings, item_ids, 'Flat')

                # Compare results
                query = test_queries[0]
                approx_results = index.search(query, k=10)
                exact_results = flat_index.search(query, k=10)

                # Overlap
                approx_ids = set(item_id for item_id, _ in approx_results)
                exact_ids = set(item_id for item_id, _ in exact_results)
                overlap = len(approx_ids & exact_ids)

                print(f"  Accuracy (top-10 overlap): {overlap}/10 = {overlap/10*100:.1f}%")
    ```

    **Expected Performance:**

    | Index Type | Build Time | Query Latency | Accuracy | Use Case |
    |------------|------------|---------------|----------|----------|
    | Flat | 1 sec | 50ms (1M items) | 100% | Baseline, small datasets |
    | IVF | 10 sec | 5ms | 90-95% | Production (balanced) |
    | HNSW | 30 sec | 2ms | 95-98% | Ultra-low latency |

    ---

    ## 4.3 Model Quantization for Faster Inference

    **Problem:** Deep learning models are slow (20-50ms) and memory-intensive.

    **Solution:** Quantize models from FP32 to INT8 for 2-4x speedup.

    ### PyTorch Model Quantization

    ```python
    import torch
    import torch.quantization
    from torch import nn
    import time

    class DeepModel(nn.Module):
        """Example deep learning model."""

        def __init__(self, embedding_dim=256, hidden_dim=512):
            super().__init__()
            self.embedding_dim = embedding_dim

            self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            self.relu = nn.ReLU()

        def forward(self, user_embedding, item_embedding):
            # Concatenate embeddings
            x = torch.cat([user_embedding, item_embedding], dim=1)

            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            score = self.fc3(x)

            return score

    class ModelQuantizer:
        """
        Quantize PyTorch models for faster inference.
        """

        @staticmethod
        def quantize_dynamic(model: nn.Module) -> nn.Module:
            """
            Dynamic quantization (weights only).
            Good for CPU inference, minimal accuracy loss.
            """
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )
            return quantized_model

        @staticmethod
        def quantize_static(model: nn.Module, calibration_data) -> nn.Module:
            """
            Static quantization (weights + activations).
            Better compression, requires calibration data.
            """
            model.eval()

            # Prepare model for quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)

            # Calibrate with representative data
            with torch.no_grad():
                for batch in calibration_data:
                    model(*batch)

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)

            return quantized_model

        @staticmethod
        def benchmark(model: nn.Module, test_data, model_name: str = "Model"):
            """
            Benchmark model inference speed and size.
            """
            model.eval()

            # Measure size
            torch.save(model.state_dict(), '/tmp/model.pth')
            import os
            size_mb = os.path.getsize('/tmp/model.pth') / (1024 * 1024)

            # Measure latency
            num_iterations = 1000
            start = time.time()

            with torch.no_grad():
                for batch in test_data[:num_iterations]:
                    model(*batch)

            elapsed = time.time() - start
            latency_ms = (elapsed / num_iterations) * 1000

            print(f"{model_name}:")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Latency: {latency_ms:.2f} ms/inference")
            print(f"  Throughput: {1000/latency_ms:.1f} inferences/sec")

            return {'size_mb': size_mb, 'latency_ms': latency_ms}


    # Example: Quantize model
    def quantize_prediction_model():
        """
        Demonstrate model quantization.
        """
        # Create model
        model = DeepModel(embedding_dim=256, hidden_dim=512)

        # Generate test data
        test_data = [
            (torch.randn(32, 256), torch.randn(32, 256))  # Batch of 32
            for _ in range(1000)
        ]

        # Benchmark original model
        print("\n=== Original Model (FP32) ===")
        original_stats = ModelQuantizer.benchmark(model, test_data, "Original")

        # Dynamic quantization
        print("\n=== Dynamically Quantized Model (INT8) ===")
        quantized_dynamic = ModelQuantizer.quantize_dynamic(model)
        dynamic_stats = ModelQuantizer.benchmark(quantized_dynamic, test_data, "Dynamic INT8")

        # Static quantization (requires calibration)
        print("\n=== Statically Quantized Model (INT8) ===")
        calibration_data = test_data[:100]  # Use 100 batches for calibration
        quantized_static = ModelQuantizer.quantize_static(model, calibration_data)
        static_stats = ModelQuantizer.benchmark(quantized_static, test_data, "Static INT8")

        # Compare
        print("\n=== Comparison ===")
        print(f"Dynamic quantization:")
        print(f"  Size reduction: {(1 - dynamic_stats['size_mb'] / original_stats['size_mb']) * 100:.1f}%")
        print(f"  Speedup: {original_stats['latency_ms'] / dynamic_stats['latency_ms']:.2f}x")

        print(f"Static quantization:")
        print(f"  Size reduction: {(1 - static_stats['size_mb'] / original_stats['size_mb']) * 100:.1f}%")
        print(f"  Speedup: {original_stats['latency_ms'] / static_stats['latency_ms']:.2f}x")
    ```

    **Expected Results:**

    - **Size reduction:** 60-75% (4x smaller)
    - **Speedup:** 2-4x faster inference
    - **Accuracy loss:** <1% for dynamic, <3% for static

    ---

    ## 4.4 Geo-Distributed Serving

    **Problem:** High latency for global users (e.g., US users accessing EU servers).

    **Solution:** Deploy prediction services in multiple regions with edge caching.

    ### Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Global Traffic Manager                      │
    │              (Route 53, CloudFlare, Akamai GTM)                 │
    └─────────────────────────────────────────────────────────────────┘
                │           │           │           │
                ▼           ▼           ▼           ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ US-East  │  │ US-West  │  │   EU     │  │  APAC    │
        │ Region   │  │ Region   │  │ Region   │  │ Region   │
        └──────────┘  └──────────┘  └──────────┘  └──────────┘
             │             │             │             │
             ▼             ▼             ▼             ▼
        Edge Cache    Edge Cache    Edge Cache    Edge Cache
        (CloudFront)  (CloudFront)  (CloudFront)  (CloudFront)
             │             │             │             │
             ▼             ▼             ▼             ▼
        Prediction    Prediction    Prediction    Prediction
        Service       Service       Service       Service
             │             │             │             │
             └─────────────┴─────────────┴─────────────┘
                          │
                          ▼
                  Central Feature Store
                  (Global Redis Cluster)
                          │
                          ▼
                  Model Registry (S3/GCS)
    ```

    ### Geo-Routing Logic

    ```python
    from dataclasses import dataclass
    from typing import Dict, Optional
    import requests

    @dataclass
    class RegionEndpoint:
        """Regional prediction service endpoint."""
        region: str
        endpoint_url: str
        latency_ms: float  # Expected latency
        capacity_qps: int  # Max QPS

    class GeoDistributedPredictionService:
        """
        Route requests to nearest region with fallback.
        """

        def __init__(self):
            self.regions = {
                'us-east-1': RegionEndpoint('us-east-1', 'https://pred-use1.example.com', 20, 20000),
                'us-west-2': RegionEndpoint('us-west-2', 'https://pred-usw2.example.com', 25, 15000),
                'eu-west-1': RegionEndpoint('eu-west-1', 'https://pred-euw1.example.com', 30, 10000),
                'ap-southeast-1': RegionEndpoint('ap-southeast-1', 'https://pred-apse1.example.com', 35, 8000)
            }

            # Region affinity (route to preferred region based on user location)
            self.region_affinity = {
                'US': 'us-east-1',
                'CA': 'us-west-2',
                'UK': 'eu-west-1',
                'DE': 'eu-west-1',
                'SG': 'ap-southeast-1',
                'JP': 'ap-southeast-1'
            }

        def route_request(self, user_id: str, context: Dict,
                         user_country: str) -> Dict:
            """
            Route prediction request to optimal region.
            """
            # Determine primary region
            primary_region = self.region_affinity.get(user_country, 'us-east-1')

            # Try primary region
            try:
                result = self._call_region(primary_region, user_id, context, timeout=0.1)
                result['region'] = primary_region
                return result
            except Exception as e:
                print(f"Primary region {primary_region} failed: {e}")

            # Fallback to nearest available region
            fallback_regions = self._get_fallback_regions(primary_region)

            for region in fallback_regions:
                try:
                    result = self._call_region(region, user_id, context, timeout=0.15)
                    result['region'] = region
                    result['fallback'] = True
                    return result
                except Exception as e:
                    print(f"Fallback region {region} failed: {e}")

            # All regions failed, return error
            raise Exception("All regions unavailable")

        def _call_region(self, region: str, user_id: str,
                        context: Dict, timeout: float) -> Dict:
            """
            Call prediction service in specific region.
            """
            endpoint = self.regions[region]

            response = requests.post(
                f"{endpoint.endpoint_url}/v1/predict",
                json={'user_id': user_id, 'context': context},
                timeout=timeout
            )

            response.raise_for_status()
            return response.json()

        def _get_fallback_regions(self, primary_region: str) -> list:
            """
            Get fallback regions in order of preference.
            """
            # Define fallback chain
            fallback_chains = {
                'us-east-1': ['us-west-2', 'eu-west-1'],
                'us-west-2': ['us-east-1', 'ap-southeast-1'],
                'eu-west-1': ['us-east-1', 'us-west-2'],
                'ap-southeast-1': ['us-west-2', 'eu-west-1']
            }

            return fallback_chains.get(primary_region, list(self.regions.keys()))
    ```

    ### Edge Caching with CloudFront

    ```python
    import boto3
    from typing import Dict

    class EdgeCacheManager:
        """
        Manage edge caching for predictions (CloudFront/CDN).
        """

        def __init__(self):
            self.cloudfront = boto3.client('cloudfront')

        def cache_predictions(self, user_id: str, predictions: Dict,
                            cache_ttl: int = 3600):
            """
            Cache predictions at edge locations.
            Uses CloudFront with custom cache key.
            """
            cache_key = f"pred:{user_id}"

            # Set cache headers
            headers = {
                'Cache-Control': f'public, max-age={cache_ttl}',
                'X-Cache-Key': cache_key,
                'Content-Type': 'application/json'
            }

            return headers

        def invalidate_cache(self, user_ids: list):
            """
            Invalidate edge cache for specific users (e.g., after new data).
            """
            paths = [f"/v1/predict?user_id={uid}*" for uid in user_ids]

            self.cloudfront.create_invalidation(
                DistributionId='DISTRIBUTION_ID',
                InvalidationBatch={
                    'Paths': {'Quantity': len(paths), 'Items': paths},
                    'CallerReference': str(time.time())
                }
            )
    ```

    ---

    ## 4.5 Batch Processing Optimization

    **Problem:** Batch precomputation takes 4-6 hours for 2B predictions.

    **Solution:** Optimize with Spark tuning and GPU-accelerated inference.

    ### Spark Optimization

    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
    import pandas as pd
    import numpy as np

    class OptimizedBatchPrediction:
        """
        GPU-accelerated batch prediction with Spark.
        """

        def __init__(self):
            # Configure Spark for performance
            self.spark = SparkSession.builder \
                .appName("BatchPrediction") \
                .config("spark.sql.shuffle.partitions", "2000") \
                .config("spark.default.parallelism", "2000") \
                .config("spark.executor.instances", "200") \
                .config("spark.executor.cores", "8") \
                .config("spark.executor.memory", "32g") \
                .config("spark.dynamicAllocation.enabled", "true") \
                .getOrCreate()

        def batch_predict(self, users_path: str, items_path: str,
                         model_path: str, output_path: str):
            """
            Batch predict for all user-item pairs.
            """
            # Load data
            users = self.spark.read.parquet(users_path)  # 100M users
            items = self.spark.read.parquet(items_path)  # 10M items

            # Filter to hot users (reduce combinations)
            hot_users = users.filter("activity_score > 0.5").limit(20_000_000)  # 20M users
            hot_items = items.filter("popularity > 0.7").limit(1_000_000)  # 1M items

            # Cross join (20M × 1M = 20B pairs, too large)
            # Instead: Join on category or use candidate generation

            # Option 1: Join on category (reduces pairs)
            user_categories = hot_users.select("user_id", "favorite_categories")
            item_categories = hot_items.select("item_id", "category")

            # Explode user categories and join
            from pyspark.sql.functions import explode
            user_cat_exploded = user_categories.select(
                "user_id", explode("favorite_categories").alias("category")
            )

            # Join users with items in their favorite categories
            pairs = user_cat_exploded.join(item_categories, on="category")

            # Now pairs is manageable (e.g., 2B pairs)
            print(f"Generated {pairs.count()} user-item pairs")

            # Load model (broadcast to all executors)
            model = self._load_model(model_path)
            broadcast_model = self.spark.sparkContext.broadcast(model)

            # Define UDF for batch prediction
            @pandas_udf("score float", PandasUDFType.SCALAR)
            def predict_udf(user_features: pd.Series, item_features: pd.Series) -> pd.Series:
                """
                Vectorized prediction UDF (runs on executor).
                """
                model = broadcast_model.value

                # Convert to numpy
                user_feat = np.array(user_features.tolist())
                item_feat = np.array(item_features.tolist())

                # Batch inference
                scores = model.predict(user_feat, item_feat)

                return pd.Series(scores)

            # Apply prediction
            predictions = pairs.withColumn(
                "score", predict_udf("user_features", "item_features")
            )

            # Filter top predictions per user
            from pyspark.sql.window import Window
            from pyspark.sql.functions import row_number

            window = Window.partitionBy("user_id").orderBy(col("score").desc())
            top_predictions = predictions.withColumn("rank", row_number().over(window)) \
                .filter(col("rank") <= 100)  # Keep top 100 per user

            # Write to output (Parquet)
            top_predictions.write.mode("overwrite").parquet(output_path)

            print(f"Wrote predictions to {output_path}")

        def _load_model(self, model_path: str):
            """Load model from S3/HDFS."""
            # Implementation depends on model format
            pass
    ```

    **Performance Improvements:**

    | Optimization | Speedup | Cost |
    |--------------|---------|------|
    | Partition pruning (filter hot users/items) | 5x | Reduced coverage |
    | Join on category (vs cross join) | 10x | Fewer pairs |
    | Vectorized UDF (Pandas UDF) | 3x | - |
    | Dynamic allocation (scale executors) | 2x | - |
    | GPU-accelerated inference | 10x | GPU cost |

    **Total improvement:** 2 hours instead of 6 hours (3x faster).

---

## Trade-offs & Design Decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| **Feature Freshness** | Real-time (1s) | Batch (1 hour) | Hybrid | Real-time for hot features, batch for historical |
| **Caching Strategy** | Single-level (predictions) | Multi-level (pred + features) | Multi-level | 90% hit rate vs 60% with single level |
| **Model Serving** | CPU-only | GPU + CPU | Hybrid | CPU for tree models (80%), GPU for deep (20%) |
| **Candidate Generation** | Exact search | Approximate (ANN) | ANN | 10x faster with <5% accuracy loss |
| **Geo-Distribution** | Single region | Multi-region | Multi-region | 50-100ms latency reduction for global users |
| **Fallback Strategy** | Fail fast | Multi-level fallback | Multi-level | 99.99% availability vs 99.9% |

---

## Real-World Examples

### Amazon Personalize Architecture

**Key Features:**

1. **Real-time event ingestion:** User events (clicks, purchases) streamed via Kinesis
2. **Online feature store:** DynamoDB for low-latency feature retrieval (<5ms)
3. **AutoML model selection:** Automatically choose best algorithm (HRNN, SIMS, etc.)
4. **Batch recommendations:** Pre-compute recommendations for 100M+ users daily
5. **Real-time personalization:** Update recommendations based on current session
6. **A/B testing:** Built-in experimentation framework

**Scale:**

- 100B events/day
- 10ms p99 latency
- 99.99% availability

**Reference:** [Amazon Personalize Documentation](https://aws.amazon.com/personalize/)

---

### Netflix Recommendation System

**Key Features:**

1. **Multi-armed bandit:** Dynamic traffic allocation for A/B tests
2. **Three-tier architecture:**
   - **Offline:** Batch compute recommendations (Spark)
   - **Nearline:** Streaming updates (Flink, every 5 minutes)
   - **Online:** Real-time personalization (Node.js, <100ms)
3. **Feature engineering:** 500+ features per prediction
4. **Model ensembling:** Combine 10+ models per request
5. **Geo-distributed:** Deployed in 50+ AWS regions

**Scale:**

- 200M+ users
- 1B+ recommendations/day
- <100ms p99 latency

**Reference:** [Netflix Tech Blog](https://netflixtechblog.com/)

---

## Monitoring & Observability

### Key Metrics

```python
# Latency metrics
prediction_latency_p50
prediction_latency_p95
prediction_latency_p99
feature_fetch_latency
model_inference_latency

# Availability
prediction_success_rate
feature_store_availability
model_serving_availability
fallback_rate (target: <5%)

# Accuracy
online_ctr (click-through rate)
online_conversion_rate
prediction_quality_score
model_drift_score

# Cache performance
l1_cache_hit_rate (target: >50%)
l2_cache_hit_rate (target: >30%)
combined_cache_hit_rate (target: >90%)

# Resource utilization
cpu_utilization (target: 60-80%)
memory_utilization
gpu_utilization (target: >80%)
cache_memory_usage

# Business metrics
revenue_per_prediction
user_engagement_rate
session_duration
```

---

## Interview Tips

1. **Start with requirements:** Clarify latency, scale, and accuracy targets
2. **Multi-level caching is critical:** Explain L1/L2/L3 hierarchy clearly
3. **Feature freshness trade-off:** Real-time vs batch, streaming vs polling
4. **Fallback strategy:** Multiple levels, circuit breaker, graceful degradation
5. **A/B testing:** Built into architecture, not an afterthought
6. **Scale numbers:** Justify cache sizes, compute resources with calculations
7. **Real-world examples:** Reference Amazon Personalize, Netflix architecture

**Common Follow-ups:**

- How to handle cold start? → Popular items, collaborative filtering
- How to ensure feature consistency? → Versioning, timestamps
- How to debug prediction quality issues? → Logging, A/B test metrics
- How to handle model updates without downtime? → Blue-green deployment
- How to reduce costs? → Batch precomputation, cheaper fallbacks

---

## References

1. **Amazon Personalize:** [AWS Personalize](https://aws.amazon.com/personalize/)
2. **Netflix Recommendations:** [Netflix Tech Blog](https://netflixtechblog.com/)
3. **Feature Stores:** [Feast](https://feast.dev/), [Tecton](https://www.tecton.ai/)
4. **Model Serving:** [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), [TorchServe](https://pytorch.org/serve/)
5. **Embedding Search:** [FAISS](https://github.com/facebookresearch/faiss), [Annoy](https://github.com/spotify/annoy)
6. **Streaming:** [Apache Flink](https://flink.apache.org/), [Kafka Streams](https://kafka.apache.org/documentation/streams/)
7. **A/B Testing:** [Optimizely](https://www.optimizely.com/), [LaunchDarkly](https://launchdarkly.com/)

---

**Last Updated:** 2026-02-05
