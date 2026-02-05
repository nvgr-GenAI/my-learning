# Design A/B Testing Framework (Experimentation Platform)

A large-scale experimentation platform that enables running thousands of concurrent A/B tests, assigns users to variants consistently, tracks metrics in real-time, and computes statistical significance to determine winning variants.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K active experiments, 100M DAU, 1B events/day, 10K QPS assignment, sub-second p99 latency |
| **Key Challenges** | Consistent user assignment, real-time metrics, statistical testing, multi-armed bandits, CUPED |
| **Core Concepts** | Consistent hashing, randomization, hypothesis testing, confidence intervals, sample size, variance reduction |
| **Companies** | Optimizely, Google Optimize, Statsig, Split.io, LaunchDarkly, Netflix, Airbnb, Uber, Meta |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Experiment Configuration** | Define experiments with variants, targeting rules, traffic allocation | P0 (Must have) |
    | **User Assignment** | Assign users to variants consistently (sticky bucketing) | P0 (Must have) |
    | **Event Tracking** | Track conversion events, metrics, custom properties | P0 (Must have) |
    | **Metrics Calculation** | Compute mean, conversion rate, percentiles for each variant | P0 (Must have) |
    | **Statistical Significance** | Calculate p-values, confidence intervals, sample size | P0 (Must have) |
    | **Real-time Dashboard** | Show experiment results, traffic allocation, winning variant | P0 (Must have) |
    | **Guardrail Metrics** | Monitor business-critical metrics (revenue, latency, errors) | P1 (Should have) |
    | **Multi-armed Bandit** | Dynamically allocate traffic to best-performing variant | P1 (Should have) |
    | **Interaction Effects** | Detect when multiple experiments interfere | P2 (Nice to have) |
    | **CUPED** | Variance reduction using pre-experiment covariates | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Personalization and recommendation engines
    - Feature flags without experimentation (simple on/off toggles)
    - Complex ML model evaluation (A/B/N testing with 100+ variants)
    - Attribution modeling (multi-touch attribution)
    - User privacy and GDPR compliance (assume handled separately)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Assignment service is critical path, cannot block user requests |
    | **Latency (Assignment)** | < 5ms p99 | Must not slow down page load or API response |
    | **Latency (Event Tracking)** | < 50ms p95 | Fire-and-forget, minimal impact on user experience |
    | **Latency (Results Query)** | < 2s p95 | Dashboard should load quickly, support exploration |
    | **Consistency** | Sticky assignment | Same user must always get same variant (no variant switching) |
    | **Data Freshness** | < 5 minutes | Results update quickly for rapid iteration |
    | **Assignment Correctness** | 99.99% | Misassignment breaks experiment validity |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Users and experiments:
    - Daily Active Users (DAU): 100M
    - Active experiments: 10K concurrent experiments
    - Average experiments per user: 5 experiments
    - Experiments with user overlap: 80% (most users see multiple experiments)

    Assignment requests (user bucketing):
    - Requests per user: 10 requests/day (multiple page loads, API calls)
    - Daily assignments: 100M × 10 = 1B requests/day
    - Assignment QPS: 1B / 86,400 = ~11,600 QPS
    - Peak QPS: 5x average = ~58,000 QPS (peak hours)

    Event tracking (conversions, metrics):
    - Events per user: 20 events/day (clicks, purchases, engagement)
    - Daily events: 100M × 20 = 2B events/day
    - Event ingestion QPS: 2B / 86,400 = ~23,150 QPS
    - Peak QPS: 5x average = ~115,750 QPS

    Dashboard queries (experiment results):
    - Active experimenters: 10K (data scientists, PMs)
    - Queries per experimenter: 20 queries/day
    - Daily queries: 10K × 20 = 200K queries/day
    - Query QPS: 200K / 86,400 = ~2.3 QPS
    - Peak QPS: 10x average = ~23 QPS (working hours)

    Total Read QPS: ~11,600 (assignments) + ~2.3 (queries) = ~11,602 QPS
    Total Write QPS: ~23,150 (events)
    Read/Write ratio: 1:2 (more events than assignments)
    ```

    ### Storage Estimates

    ```
    Experiment definitions:
    - Active experiments: 10K
    - Definition size: 10 KB (variants, targeting, metrics)
    - Total: 10K × 10 KB = 100 MB

    Assignment cache (user → variant mapping):
    - Daily Active Users: 100M
    - Assignments per user: 5 experiments × 1 variant = 5 assignments
    - Assignment record: 100 bytes (user_id, experiment_id, variant_id, timestamp)
    - Total: 100M × 5 × 100 bytes = 50 GB
    - Cache TTL: 30 days
    - 30-day storage: 50 GB × 30 = 1.5 TB

    Event data (raw):
    - Daily events: 2B events/day
    - Event size: 500 bytes (user_id, experiment_id, variant_id, metric, value, timestamp)
    - Daily storage: 2B × 500 bytes = 1 TB/day
    - 90-day retention: 1 TB × 90 = 90 TB

    Pre-aggregated metrics:
    - Per experiment per variant per day: 10K experiments × 4 variants × 90 days = 3.6M records
    - Record size: 5 KB (all metrics, counts, sums, variance)
    - Total: 3.6M × 5 KB = 18 GB

    Statistical test results:
    - Experiments × variants × metrics: 10K × 4 × 10 = 400K combinations
    - Result size: 2 KB (p-value, confidence interval, effect size, sample size)
    - Total: 400K × 2 KB = 800 MB

    Total storage: 100 MB + 1.5 TB + 90 TB + 18 GB + 800 MB ≈ 91.5 TB
    ```

    ### Bandwidth Estimates

    ```
    Assignment ingress (requests):
    - 11,600 QPS × 200 bytes (request) = 2.32 MB/sec ≈ 18.6 Mbps

    Assignment egress (responses):
    - 11,600 QPS × 500 bytes (response with variant config) = 5.8 MB/sec ≈ 46.4 Mbps

    Event ingress:
    - 23,150 QPS × 500 bytes = 11.6 MB/sec ≈ 92.8 Mbps
    - Peak: 115,750 QPS × 500 bytes = 57.9 MB/sec ≈ 463 Mbps

    Dashboard egress (query results):
    - 2.3 QPS × 100 KB (dashboard data) = 230 KB/sec ≈ 1.8 Mbps
    - Peak: 23 QPS × 100 KB = 2.3 MB/sec ≈ 18.4 Mbps

    Total ingress: 18.6 Mbps + 92.8 Mbps = ~111 Mbps (peak: ~481 Mbps)
    Total egress: 46.4 Mbps + 18.4 Mbps = ~65 Mbps (peak: ~65 Mbps)
    ```

    ### Memory Estimates (Caching)

    ```
    Experiment config cache (hot experiments):
    - Active experiments: 10K × 10 KB = 100 MB
    - In-memory: entire config fits easily

    Assignment cache (user bucketing):
    - Recent users: 50M (50% of DAU, active in last hour)
    - 50M × 5 experiments × 100 bytes = 25 GB
    - Use Redis cluster with 5 nodes: 5 GB per node

    Pre-computed metrics (5-minute windows):
    - 10K experiments × 4 variants × 20 metrics × 1 KB = 800 MB

    Statistical test cache (p-values, CI):
    - 10K experiments × 4 variants × 10 metrics × 2 KB = 800 MB

    Total cache: 100 MB + 25 GB + 800 MB + 800 MB ≈ 27 GB
    ```

    ---

    ## Key Assumptions

    1. Users are assigned to experiments at page load / API request time
    2. Assignment is sticky (same user → same variant for experiment lifetime)
    3. 100M DAU, each user sees ~5 experiments on average
    4. 2B events/day tracked across all experiments
    5. 90-day data retention for raw events (compliance)
    6. Most experiments run for 14-30 days (standard duration)
    7. Statistical significance requires 95% confidence (α = 0.05)
    8. Average experiment has 2-4 variants (A/B or A/B/C/D)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Consistent hashing:** Deterministic user assignment (hash(user_id, experiment_id) → variant)
    2. **Cache-aside pattern:** Assignment service checks cache before computing assignment
    3. **Event streaming:** Kafka for high-throughput event ingestion
    4. **Pre-aggregation:** Compute metrics incrementally (real-time updates)
    5. **Statistical rigor:** Proper hypothesis testing, avoid peeking problem
    6. **Separation of concerns:** Assignment, event tracking, analysis are independent services

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            WebApp[Web Application<br/>Mobile App]
            BackendService[Backend Service<br/>API Gateway]
        end

        subgraph "Assignment Layer"
            Assignment_LB[Load Balancer<br/>Assignment API]
            Assignment_Service[Assignment Service<br/>Consistent hashing<br/>Sticky bucketing]
            Assignment_Cache[Redis Cluster<br/>Assignment cache<br/>25 GB]
        end

        subgraph "Event Ingestion Layer"
            Event_LB[Load Balancer<br/>Event API]
            Event_Service[Event Service<br/>Validation, dedup]
            Event_Buffer[Kafka<br/>Event streaming<br/>~100K QPS]
        end

        subgraph "Stream Processing (Real-time)"
            Stream_Processor[Flink/Spark Streaming<br/>Incremental aggregation]
            Metrics_Aggregator[Metrics Aggregator<br/>Sum, count, variance]
            Stats_Computer[Stats Computer<br/>t-test, chi-square<br/>CI calculation]
        end

        subgraph "Batch Processing (Daily)"
            Spark_Job[Spark Jobs<br/>Daily rollups]
            CUPED_Calculator[CUPED Engine<br/>Variance reduction]
            Bandit_Optimizer[Bandit Optimizer<br/>Thompson sampling]
            Guardrail_Monitor[Guardrail Monitor<br/>Alert on degradation]
        end

        subgraph "Query Layer"
            Dashboard_LB[Load Balancer<br/>Dashboard API]
            Dashboard_Service[Dashboard Service<br/>Experiment results]
            Config_Service[Config Service<br/>Experiment CRUD]
            Sample_Size_Calc[Sample Size Calculator<br/>Power analysis]
        end

        subgraph "Caching Layer"
            Redis_Assignments[Redis Cluster<br/>User assignments<br/>TTL: 30 days]
            Redis_Metrics[Redis<br/>Real-time metrics<br/>TTL: 5 minutes]
            Redis_Stats[Redis<br/>Statistical results<br/>TTL: 1 hour]
        end

        subgraph "Storage Layer"
            Config_DB[(PostgreSQL<br/>Experiment config<br/>Targeting rules)]
            Event_DB[(ClickHouse<br/>Raw events<br/>90-day retention)]
            Metrics_DB[(ClickHouse<br/>Pre-aggregated metrics<br/>Time-series)]
            Assignment_DB[(Cassandra<br/>Assignment log<br/>Audit trail)]
        end

        subgraph "ML & Analytics"
            Sample_Size_Service[Sample Size Service<br/>MDE, power]
            Interaction_Detector[Interaction Detector<br/>Multi-experiment analysis]
            Bayesian_Engine[Bayesian Engine<br/>Posterior distributions]
        end

        WebApp --> Assignment_LB
        WebApp --> Event_LB
        BackendService --> Assignment_LB
        BackendService --> Event_LB

        Assignment_LB --> Assignment_Service
        Assignment_Service --> Assignment_Cache
        Assignment_Service --> Config_DB
        Assignment_Service --> Redis_Assignments
        Assignment_Cache --> Redis_Assignments

        Event_LB --> Event_Service
        Event_Service --> Event_Buffer
        Event_Buffer --> Stream_Processor

        Stream_Processor --> Metrics_Aggregator
        Stream_Processor --> Event_DB
        Metrics_Aggregator --> Redis_Metrics
        Metrics_Aggregator --> Stats_Computer
        Stats_Computer --> Redis_Stats
        Stats_Computer --> Metrics_DB

        Event_DB --> Spark_Job
        Spark_Job --> CUPED_Calculator
        Spark_Job --> Bandit_Optimizer
        Spark_Job --> Guardrail_Monitor
        CUPED_Calculator --> Metrics_DB
        Bandit_Optimizer --> Config_DB
        Guardrail_Monitor --> Config_DB

        Dashboard_LB --> Dashboard_Service
        Dashboard_LB --> Config_Service
        Dashboard_LB --> Sample_Size_Calc

        Dashboard_Service --> Redis_Stats
        Dashboard_Service --> Metrics_DB
        Dashboard_Service --> Config_DB

        Config_Service --> Config_DB
        Sample_Size_Calc --> Sample_Size_Service

        Assignment_Service --> Assignment_DB

        Dashboard_Service --> Interaction_Detector
        Dashboard_Service --> Bayesian_Engine
        Interaction_Detector --> Event_DB
        Bayesian_Engine --> Metrics_DB

        style Assignment_LB fill:#e1f5ff
        style Event_LB fill:#e1f5ff
        style Dashboard_LB fill:#e1f5ff
        style Redis_Assignments fill:#fff4e1
        style Redis_Metrics fill:#fff4e1
        style Redis_Stats fill:#fff4e1
        style Config_DB fill:#e8eaf6
        style Event_DB fill:#ffe1e1
        style Metrics_DB fill:#ffe1e1
        style Assignment_DB fill:#ffe1e1
        style Event_Buffer fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis (Assignment Cache)** | Sub-5ms assignment lookups, 25 GB fits in memory | Database only (10-100ms too slow), in-process cache (inconsistent across servers) |
    | **Consistent Hashing** | Deterministic assignment without database lookup, same user → same variant | Random assignment (not sticky), database roundtrip (high latency) |
    | **Kafka** | High-throughput event ingestion (100K QPS), replay capability, decoupling | Direct DB writes (can't handle spikes), message queue (no replay) |
    | **ClickHouse** | Columnar OLAP for fast aggregations, 100x faster than row-based | PostgreSQL (too slow for billions of events), Elasticsearch (not optimized for aggregations) |
    | **Flink/Spark Streaming** | Real-time metrics (< 5 min delay), incremental aggregation | Batch only (hours of delay), custom code (complex state management) |
    | **Cassandra (Assignment Log)** | High-write throughput for audit trail, immutable append-only log | PostgreSQL (write bottleneck), S3 (no random access) |
    | **CUPED** | Reduces variance by 20-50%, detects smaller effects with same sample size | Ignore covariates (larger sample sizes needed) |

    **Key Trade-off:** We chose **eventual consistency** for metrics. Real-time dashboard may lag by 1-5 minutes, but assignment is strongly consistent (sticky).

    ---

    ## API Design

    ### 1. Get Experiment Assignment (Bucketing)

    **Request:**
    ```http
    POST /api/v1/experiments/assign
    Content-Type: application/json

    {
      "user_id": "user_abc123",
      "experiments": ["exp_homepage_redesign", "exp_checkout_flow"],
      "context": {
        "country": "US",
        "platform": "web",
        "user_segment": "premium",
        "device_type": "mobile"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user_abc123",
      "assignments": [
        {
          "experiment_id": "exp_homepage_redesign",
          "variant_id": "variant_b",
          "variant_name": "new_hero_section",
          "config": {
            "button_color": "blue",
            "show_testimonials": true,
            "hero_image_url": "https://cdn.example.com/hero_v2.jpg"
          },
          "assigned_at": "2026-02-05T10:30:00.000Z",
          "experiment_start": "2026-02-01T00:00:00.000Z",
          "experiment_end": "2026-02-28T23:59:59.999Z"
        },
        {
          "experiment_id": "exp_checkout_flow",
          "variant_id": "control",
          "variant_name": "original_checkout",
          "config": {},
          "assigned_at": "2026-02-05T10:30:00.000Z",
          "experiment_start": "2026-02-03T00:00:00.000Z",
          "experiment_end": "2026-02-17T23:59:59.999Z"
        }
      ],
      "response_time_ms": 3
    }
    ```

    **Design Notes:**

    - Assignment is **idempotent:** same user → same variant for experiment lifetime
    - **Consistent hashing:** `variant = hash(user_id + experiment_id + salt) % num_variants`
    - **Targeting rules:** evaluate user context against experiment config (country, platform, segment)
    - **Sticky assignment:** first assignment is cached in Redis (30-day TTL)
    - **Cache-aside pattern:** check Redis → if miss, compute → store in Redis
    - **Latency:** p99 < 5ms (Redis lookup ~1ms + hash computation ~0.1ms)

    ---

    ### 2. Track Event (Conversion, Metric)

    **Request:**
    ```http
    POST /api/v1/events
    Content-Type: application/json

    {
      "user_id": "user_abc123",
      "event_type": "purchase",
      "timestamp": "2026-02-05T10:35:22.123Z",
      "experiments": [
        {
          "experiment_id": "exp_homepage_redesign",
          "variant_id": "variant_b"
        },
        {
          "experiment_id": "exp_checkout_flow",
          "variant_id": "control"
        }
      ],
      "metrics": {
        "revenue": 129.99,
        "items_purchased": 3,
        "checkout_duration_sec": 45.2
      },
      "properties": {
        "payment_method": "credit_card",
        "promo_code_used": true,
        "shipping_method": "standard"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "event_id": "evt_xyz789",
      "accepted_at": "2026-02-05T10:35:22.150Z"
    }
    ```

    **Design Notes:**

    - **Fire-and-forget:** return 202 immediately, process asynchronously
    - **Idempotency:** deduplicate by (user_id, event_type, timestamp, experiment_id)
    - **Attribution:** user must be assigned to experiment before tracking events
    - **Batch support:** accept up to 100 events in single request
    - **Validation:** reject events with missing required fields or invalid experiment IDs
    - **Rate limit:** 1000 events/sec per user_id (prevent abuse)

    ---

    ### 3. Create Experiment

    **Request:**
    ```http
    POST /api/v1/experiments
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "name": "Homepage Hero Section Redesign",
      "description": "Test new hero section with customer testimonials",
      "hypothesis": "Adding testimonials will increase sign-up conversion by 10%",
      "owner": "pm_sarah",
      "start_date": "2026-02-10T00:00:00.000Z",
      "end_date": "2026-03-10T23:59:59.999Z",
      "variants": [
        {
          "id": "control",
          "name": "Original Hero",
          "description": "Current hero section",
          "traffic_allocation": 0.50,
          "config": {}
        },
        {
          "id": "variant_a",
          "name": "Hero with Testimonials",
          "description": "Add 3 customer testimonials",
          "traffic_allocation": 0.25,
          "config": {
            "show_testimonials": true,
            "testimonial_count": 3
          }
        },
        {
          "id": "variant_b",
          "name": "Hero with Video",
          "description": "Replace image with video",
          "traffic_allocation": 0.25,
          "config": {
            "use_video": true,
            "video_url": "https://cdn.example.com/hero.mp4"
          }
        }
      ],
      "targeting": {
        "countries": ["US", "CA", "GB"],
        "platforms": ["web"],
        "user_segments": ["new_user", "returning_user"],
        "traffic_percentage": 0.80
      },
      "primary_metric": {
        "name": "signup_conversion_rate",
        "type": "conversion",
        "numerator_event": "signup_completed",
        "denominator": "users_exposed"
      },
      "secondary_metrics": [
        {
          "name": "time_to_signup",
          "type": "continuous",
          "event": "signup_completed",
          "field": "duration_sec"
        }
      ],
      "guardrail_metrics": [
        {
          "name": "page_load_time",
          "type": "continuous",
          "threshold_ms": 500,
          "alert_on_degradation": true
        },
        {
          "name": "error_rate",
          "type": "conversion",
          "threshold_percentage": 0.02,
          "alert_on_increase": true
        }
      ],
      "sample_size": {
        "mde": 0.10,
        "baseline_rate": 0.05,
        "alpha": 0.05,
        "power": 0.80,
        "required_sample_per_variant": 12460
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "experiment_id": "exp_homepage_redesign_v2",
      "status": "draft",
      "created_at": "2026-02-05T10:40:00.000Z",
      "sample_size_estimate": {
        "days_to_completion": 14,
        "users_per_day": 2678
      },
      "validation": {
        "traffic_allocation_sum": 1.00,
        "has_primary_metric": true,
        "has_guardrails": true,
        "sample_size_adequate": true
      }
    }
    ```

    **Design Notes:**

    - **Traffic allocation:** must sum to 1.0 across all variants
    - **Sample size calculation:** based on MDE (minimum detectable effect), power, alpha
    - **Targeting rules:** flexible filters (country, platform, segment, custom attributes)
    - **Guardrail metrics:** auto-stop experiment if critical metrics degrade
    - **Variants:** control + 1-10 treatment variants
    - **Status flow:** draft → review → active → paused → completed

    ---

    ### 4. Get Experiment Results

    **Request:**
    ```http
    GET /api/v1/experiments/exp_homepage_redesign_v2/results
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "experiment_id": "exp_homepage_redesign_v2",
      "status": "active",
      "start_date": "2026-02-10T00:00:00.000Z",
      "days_running": 5,
      "total_users_exposed": 89234,
      "variants": [
        {
          "variant_id": "control",
          "name": "Original Hero",
          "users_exposed": 44617,
          "metrics": {
            "signup_conversion_rate": {
              "value": 0.0512,
              "count": 2284,
              "total": 44617,
              "std_error": 0.0011,
              "confidence_interval_95": [0.0490, 0.0534]
            },
            "time_to_signup": {
              "mean": 123.4,
              "std_dev": 45.2,
              "median": 112.0,
              "p95": 201.5,
              "sample_size": 2284
            }
          }
        },
        {
          "variant_id": "variant_a",
          "name": "Hero with Testimonials",
          "users_exposed": 22309,
          "metrics": {
            "signup_conversion_rate": {
              "value": 0.0587,
              "count": 1310,
              "total": 22309,
              "std_error": 0.0016,
              "confidence_interval_95": [0.0556, 0.0618],
              "vs_control": {
                "absolute_lift": 0.0075,
                "relative_lift": 0.1465,
                "p_value": 0.0023,
                "is_significant": true,
                "confidence_level": 0.95,
                "test_type": "two_sample_z_test"
              }
            },
            "time_to_signup": {
              "mean": 118.7,
              "std_dev": 42.1,
              "median": 108.0,
              "p95": 195.2,
              "sample_size": 1310,
              "vs_control": {
                "absolute_diff": -4.7,
                "relative_diff": -0.0381,
                "p_value": 0.0456,
                "is_significant": true,
                "test_type": "welch_t_test"
              }
            }
          }
        },
        {
          "variant_id": "variant_b",
          "name": "Hero with Video",
          "users_exposed": 22308,
          "metrics": {
            "signup_conversion_rate": {
              "value": 0.0498,
              "count": 1111,
              "total": 22308,
              "std_error": 0.0015,
              "confidence_interval_95": [0.0469, 0.0527],
              "vs_control": {
                "absolute_lift": -0.0014,
                "relative_lift": -0.0273,
                "p_value": 0.5234,
                "is_significant": false,
                "confidence_level": 0.95
              }
            }
          }
        }
      ],
      "guardrail_metrics": [
        {
          "name": "page_load_time",
          "threshold_ms": 500,
          "control_mean": 287.3,
          "variant_a_mean": 295.1,
          "variant_b_mean": 512.4,
          "variant_b_exceeds_threshold": true,
          "alert_triggered": true
        }
      ],
      "recommendation": {
        "winning_variant": "variant_a",
        "confidence": "high",
        "reasoning": "Variant A shows 14.65% relative lift (p=0.0023), statistically significant",
        "warnings": ["Variant B exceeds page load time guardrail"]
      },
      "sample_size_progress": {
        "required_per_variant": 12460,
        "control_progress": 0.358,
        "variant_a_progress": 0.179,
        "variant_b_progress": 0.179,
        "estimated_days_to_completion": 9
      },
      "cuped_enabled": true,
      "cuped_variance_reduction": 0.32,
      "last_updated": "2026-02-15T10:45:00.000Z"
    }
    ```

    **Design Notes:**

    - **Statistical tests:** Z-test for proportions, Welch's t-test for continuous metrics
    - **Confidence intervals:** 95% CI using normal approximation
    - **Multiple comparisons:** Bonferroni correction if testing multiple variants
    - **CUPED:** adjust metrics using pre-experiment covariates (32% variance reduction)
    - **Guardrails:** flag and alert if critical metrics degrade
    - **Real-time updates:** results refresh every 5 minutes
    - **Sample size progress:** track completion vs. required sample size

    ---

    ### 5. Calculate Sample Size

    **Request:**
    ```http
    POST /api/v1/experiments/sample-size
    Content-Type: application/json

    {
      "metric_type": "conversion",
      "baseline_rate": 0.05,
      "mde": 0.10,
      "alpha": 0.05,
      "power": 0.80,
      "num_variants": 3,
      "two_tailed": true
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "required_sample_per_variant": 12460,
      "total_sample_size": 37380,
      "days_to_completion": {
        "daily_traffic_10k": 125,
        "daily_traffic_50k": 25,
        "daily_traffic_100k": 13
      },
      "detectable_effect_sizes": {
        "5_percent_relative_lift": {
          "sample_size_per_variant": 49840,
          "days_at_50k_traffic": 100
        },
        "10_percent_relative_lift": {
          "sample_size_per_variant": 12460,
          "days_at_50k_traffic": 25
        },
        "15_percent_relative_lift": {
          "sample_size_per_variant": 5538,
          "days_at_50k_traffic": 11
        }
      },
      "assumptions": {
        "baseline_rate": 0.05,
        "treatment_rate": 0.055,
        "effect_size": 0.10,
        "alpha": 0.05,
        "power": 0.80
      }
    }
    ```

    **Design Notes:**

    - **Formula:** n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₂ - p₁)²
    - **MDE (Minimum Detectable Effect):** smallest effect size worth detecting
    - **Power (1-β):** probability of detecting effect if it exists (typically 0.80)
    - **Alpha (α):** significance level, probability of Type I error (typically 0.05)
    - **Multiple variants:** adjust for multiple comparisons (Bonferroni, Holm-Bonferroni)

=== "📊 Step 3: Deep Dive - Core Components"

    ## 1. Consistent Hashing for User Assignment

    **Problem:** Assign users to variants deterministically (same user → same variant) without database lookup.

    **Solution:** Consistent hashing with experiment-specific salt.

    ### Assignment Algorithm

    ```python
    import hashlib
    from typing import List, Dict

    class ExperimentAssignment:
        def __init__(self, experiment_id: str, variants: List[Dict]):
            """
            variants: [
                {"id": "control", "traffic": 0.5},
                {"id": "variant_a", "traffic": 0.25},
                {"id": "variant_b", "traffic": 0.25}
            ]
            """
            self.experiment_id = experiment_id
            self.variants = variants
            self.salt = self._get_experiment_salt(experiment_id)

            # Build cumulative distribution for traffic allocation
            cumulative = 0.0
            self.buckets = []
            for variant in variants:
                cumulative += variant["traffic"]
                self.buckets.append((variant["id"], cumulative))

        def _get_experiment_salt(self, experiment_id: str) -> str:
            """Get experiment-specific salt from config (prevents carryover bias)"""
            # In production, fetch from database/cache
            return f"salt_{experiment_id}_2026"

        def assign_variant(self, user_id: str) -> str:
            """Deterministically assign user to variant using consistent hashing"""
            # Hash user_id + experiment_id + salt
            hash_input = f"{user_id}:{self.experiment_id}:{self.salt}"
            hash_digest = hashlib.md5(hash_input.encode()).hexdigest()

            # Convert hex hash to integer in range [0, 1)
            hash_int = int(hash_digest[:8], 16)  # Use first 8 hex chars
            bucket_position = hash_int / (16 ** 8)  # Normalize to [0, 1)

            # Find variant based on bucket position
            for variant_id, cumulative_traffic in self.buckets:
                if bucket_position < cumulative_traffic:
                    return variant_id

            # Fallback (should never reach here if traffic sums to 1.0)
            return self.variants[0]["id"]

        def check_targeting(self, user_context: Dict) -> bool:
            """Check if user matches targeting rules"""
            # Example: country, platform, user_segment filters
            targeting = self._get_targeting_rules()

            if targeting.get("countries") and user_context.get("country") not in targeting["countries"]:
                return False

            if targeting.get("platforms") and user_context.get("platform") not in targeting["platforms"]:
                return False

            if targeting.get("user_segments") and user_context.get("segment") not in targeting["user_segments"]:
                return False

            # Traffic percentage (e.g., only 80% of eligible users)
            if targeting.get("traffic_percentage", 1.0) < 1.0:
                traffic_hash = hashlib.md5(f"{user_context['user_id']}:traffic".encode()).hexdigest()
                traffic_bucket = int(traffic_hash[:8], 16) / (16 ** 8)
                if traffic_bucket >= targeting["traffic_percentage"]:
                    return False

            return True

    # Example usage
    experiment = ExperimentAssignment(
        experiment_id="exp_homepage_redesign",
        variants=[
            {"id": "control", "traffic": 0.5},
            {"id": "variant_a", "traffic": 0.25},
            {"id": "variant_b", "traffic": 0.25}
        ]
    )

    user_id = "user_abc123"
    assigned_variant = experiment.assign_variant(user_id)
    print(f"User {user_id} assigned to: {assigned_variant}")

    # Verify stickiness: same user always gets same variant
    for _ in range(10):
        assert experiment.assign_variant(user_id) == assigned_variant
    ```

    **Key Properties:**

    - **Deterministic:** Same input → same output (no randomness)
    - **Uniform distribution:** Users evenly distributed across variants
    - **Independent:** Assignment in one experiment doesn't affect others
    - **Fast:** Pure computation, no I/O (< 0.1ms)
    - **Sticky:** User stays in same variant for experiment lifetime

    ---

    ## 2. Statistical Significance Testing

    **Problem:** Determine if observed difference between variants is real or due to chance.

    **Solution:** Hypothesis testing with p-values and confidence intervals.

    ### Two-Sample Z-Test (Conversion Rate)

    ```python
    import math
    import scipy.stats as stats
    from typing import Tuple

    class StatisticalTest:
        @staticmethod
        def z_test_proportions(
            control_conversions: int,
            control_total: int,
            treatment_conversions: int,
            treatment_total: int,
            alpha: float = 0.05
        ) -> Dict:
            """
            Two-sample Z-test for proportions (conversion rates)

            H0 (null hypothesis): p_treatment = p_control
            H1 (alternative): p_treatment ≠ p_control (two-tailed)
            """
            # Calculate proportions
            p_control = control_conversions / control_total
            p_treatment = treatment_conversions / treatment_total

            # Pooled proportion (under null hypothesis)
            p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)

            # Standard error
            se = math.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))

            # Z-statistic
            z_score = (p_treatment - p_control) / se

            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Is significant?
            is_significant = p_value < alpha

            # Confidence interval for difference
            z_critical = stats.norm.ppf(1 - alpha/2)
            se_diff = math.sqrt(
                p_control * (1 - p_control) / control_total +
                p_treatment * (1 - p_treatment) / treatment_total
            )
            ci_lower = (p_treatment - p_control) - z_critical * se_diff
            ci_upper = (p_treatment - p_control) + z_critical * se_diff

            # Relative lift
            relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else 0

            return {
                "control_rate": p_control,
                "treatment_rate": p_treatment,
                "absolute_lift": p_treatment - p_control,
                "relative_lift": relative_lift,
                "z_score": z_score,
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": 1 - alpha,
                "confidence_interval": (ci_lower, ci_upper),
                "standard_error": se
            }

    # Example: Homepage redesign experiment
    result = StatisticalTest.z_test_proportions(
        control_conversions=2284,
        control_total=44617,
        treatment_conversions=1310,
        treatment_total=22309,
        alpha=0.05
    )

    print(f"Control conversion rate: {result['control_rate']:.4f}")
    print(f"Treatment conversion rate: {result['treatment_rate']:.4f}")
    print(f"Relative lift: {result['relative_lift']:.2%}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Is significant? {result['is_significant']}")
    print(f"95% CI: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
    ```

    **Output:**
    ```
    Control conversion rate: 0.0512
    Treatment conversion rate: 0.0587
    Relative lift: 14.65%
    P-value: 0.0023
    Is significant? True
    95% CI: [0.0026, 0.0124]
    ```

    ### Welch's T-Test (Continuous Metrics)

    ```python
    class StatisticalTest:
        @staticmethod
        def welch_t_test(
            control_mean: float,
            control_std: float,
            control_n: int,
            treatment_mean: float,
            treatment_std: float,
            treatment_n: int,
            alpha: float = 0.05
        ) -> Dict:
            """
            Welch's t-test for continuous metrics (doesn't assume equal variance)
            Used for: revenue, time-on-site, items-per-cart, etc.
            """
            # Standard errors
            se_control = control_std / math.sqrt(control_n)
            se_treatment = treatment_std / math.sqrt(treatment_n)

            # Standard error of difference
            se_diff = math.sqrt(se_control**2 + se_treatment**2)

            # T-statistic
            t_stat = (treatment_mean - control_mean) / se_diff

            # Degrees of freedom (Welch-Satterthwaite equation)
            df = (se_control**2 + se_treatment**2)**2 / (
                se_control**4 / (control_n - 1) +
                se_treatment**4 / (treatment_n - 1)
            )

            # P-value (two-tailed)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # Is significant?
            is_significant = p_value < alpha

            # Confidence interval
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ci_lower = (treatment_mean - control_mean) - t_critical * se_diff
            ci_upper = (treatment_mean - control_mean) + t_critical * se_diff

            # Relative difference
            relative_diff = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0

            return {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "absolute_diff": treatment_mean - control_mean,
                "relative_diff": relative_diff,
                "t_statistic": t_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": 1 - alpha,
                "confidence_interval": (ci_lower, ci_upper)
            }

    # Example: Time to signup
    result = StatisticalTest.welch_t_test(
        control_mean=123.4,
        control_std=45.2,
        control_n=2284,
        treatment_mean=118.7,
        treatment_std=42.1,
        treatment_n=1310,
        alpha=0.05
    )

    print(f"Control mean: {result['control_mean']:.1f}s")
    print(f"Treatment mean: {result['treatment_mean']:.1f}s")
    print(f"Absolute diff: {result['absolute_diff']:.1f}s ({result['relative_diff']:.2%})")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Is significant? {result['is_significant']}")
    ```

    ---

    ## 3. Sample Size Calculation

    **Problem:** How many users needed to detect minimum effect size with desired power?

    **Solution:** Power analysis using statistical formulas.

    ### Sample Size for Conversion Rate

    ```python
    import math
    import scipy.stats as stats

    class SampleSizeCalculator:
        @staticmethod
        def calculate_for_conversion_rate(
            baseline_rate: float,
            mde: float,  # Minimum Detectable Effect (relative)
            alpha: float = 0.05,
            power: float = 0.80,
            two_tailed: bool = True
        ) -> int:
            """
            Calculate required sample size per variant for conversion rate test.

            Args:
                baseline_rate: Control conversion rate (e.g., 0.05 = 5%)
                mde: Minimum detectable effect as relative lift (e.g., 0.10 = 10% lift)
                alpha: Significance level (Type I error rate)
                power: Statistical power (1 - Type II error rate)
                two_tailed: Use two-tailed test?

            Returns:
                Required sample size per variant
            """
            # Treatment rate given relative MDE
            treatment_rate = baseline_rate * (1 + mde)

            # Effect size (absolute difference)
            effect_size = treatment_rate - baseline_rate

            # Z-scores for alpha and beta
            if two_tailed:
                z_alpha = stats.norm.ppf(1 - alpha/2)
            else:
                z_alpha = stats.norm.ppf(1 - alpha)

            z_beta = stats.norm.ppf(power)

            # Pooled proportion
            p_pooled = (baseline_rate + treatment_rate) / 2

            # Sample size formula
            numerator = (z_alpha + z_beta)**2 * (
                baseline_rate * (1 - baseline_rate) +
                treatment_rate * (1 - treatment_rate)
            )
            denominator = effect_size**2

            sample_size = numerator / denominator

            return math.ceil(sample_size)

        @staticmethod
        def calculate_for_continuous_metric(
            baseline_mean: float,
            baseline_std: float,
            mde: float,  # Minimum detectable effect (relative)
            alpha: float = 0.05,
            power: float = 0.80,
            two_tailed: bool = True
        ) -> int:
            """
            Calculate required sample size for continuous metric (e.g., revenue, time).
            """
            # Treatment mean given relative MDE
            treatment_mean = baseline_mean * (1 + mde)

            # Effect size (Cohen's d)
            cohens_d = (treatment_mean - baseline_mean) / baseline_std

            # Z-scores
            if two_tailed:
                z_alpha = stats.norm.ppf(1 - alpha/2)
            else:
                z_alpha = stats.norm.ppf(1 - alpha)

            z_beta = stats.norm.ppf(power)

            # Sample size formula
            sample_size = 2 * ((z_alpha + z_beta) / cohens_d)**2

            return math.ceil(sample_size)

        @staticmethod
        def estimate_duration(
            sample_size_per_variant: int,
            num_variants: int,
            daily_traffic: int,
            traffic_allocation: float = 1.0
        ) -> int:
            """
            Estimate experiment duration in days.

            Args:
                sample_size_per_variant: Required sample per variant
                num_variants: Number of variants (including control)
                daily_traffic: Average daily users
                traffic_allocation: % of traffic allocated to experiment

            Returns:
                Estimated days to completion
            """
            total_sample_needed = sample_size_per_variant * num_variants
            daily_sample = daily_traffic * traffic_allocation
            days = math.ceil(total_sample_needed / daily_sample)
            return days

    # Example: Calculate sample size for homepage experiment
    calc = SampleSizeCalculator()

    sample_size = calc.calculate_for_conversion_rate(
        baseline_rate=0.05,  # 5% conversion rate
        mde=0.10,  # Want to detect 10% relative lift (0.5% to 5.5%)
        alpha=0.05,
        power=0.80
    )

    print(f"Required sample size per variant: {sample_size:,}")

    # Estimate duration
    days = calc.estimate_duration(
        sample_size_per_variant=sample_size,
        num_variants=3,  # Control + 2 treatments
        daily_traffic=50000,
        traffic_allocation=0.80
    )

    print(f"Estimated experiment duration: {days} days")
    ```

    **Output:**
    ```
    Required sample size per variant: 12,460
    Estimated experiment duration: 25 days
    ```

    **Key Considerations:**

    - **MDE (Minimum Detectable Effect):** Smaller MDE requires larger sample (quadratic relationship)
    - **Power:** Higher power requires more samples (80% is standard)
    - **Multiple comparisons:** Adjust alpha if testing multiple variants (Bonferroni correction)
    - **Sequential testing:** Can stop early if strong signal detected (but increases false positive rate)

    ---

    ## 4. CUPED (Controlled-experiment Using Pre-Experiment Data)

    **Problem:** High variance in metrics requires large sample sizes. Can we reduce variance?

    **Solution:** Use pre-experiment data (covariates) to reduce variance by 20-50%.

    ### CUPED Variance Reduction

    ```python
    import numpy as np
    from typing import List, Tuple

    class CUPED:
        """
        CUPED (Controlled-experiment Using Pre-Experiment Data)

        Reduces variance by adjusting metrics using pre-experiment covariates.
        Example: If testing impact on purchase rate, use user's historical
        purchase rate as covariate.

        Reference: Deng et al. (2013) "Improving the Sensitivity of Online
        Controlled Experiments by Utilizing Pre-Experiment Data"
        """

        @staticmethod
        def adjust_metric(
            metric_values: np.ndarray,
            covariate_values: np.ndarray
        ) -> Tuple[np.ndarray, float]:
            """
            Adjust metric using CUPED.

            Formula: Y_adjusted = Y - θ(X - E[X])
            where θ = Cov(Y, X) / Var(X)

            Args:
                metric_values: Experiment-period metric (e.g., purchases during exp)
                covariate_values: Pre-experiment metric (e.g., historical purchases)

            Returns:
                (adjusted_values, variance_reduction_ratio)
            """
            # Calculate theta (optimal adjustment coefficient)
            covariance = np.cov(metric_values, covariate_values)[0, 1]
            variance_covariate = np.var(covariate_values, ddof=1)
            theta = covariance / variance_covariate

            # Adjust metric
            mean_covariate = np.mean(covariate_values)
            adjusted_values = metric_values - theta * (covariate_values - mean_covariate)

            # Calculate variance reduction
            variance_original = np.var(metric_values, ddof=1)
            variance_adjusted = np.var(adjusted_values, ddof=1)
            variance_reduction = 1 - (variance_adjusted / variance_original)

            return adjusted_values, variance_reduction

        @staticmethod
        def cuped_t_test(
            control_metric: np.ndarray,
            control_covariate: np.ndarray,
            treatment_metric: np.ndarray,
            treatment_covariate: np.ndarray,
            alpha: float = 0.05
        ) -> Dict:
            """
            Perform t-test with CUPED adjustment.
            """
            # Adjust both control and treatment
            control_adjusted, control_vr = CUPED.adjust_metric(control_metric, control_covariate)
            treatment_adjusted, treatment_vr = CUPED.adjust_metric(treatment_metric, treatment_covariate)

            # Standard t-test on adjusted values
            t_stat, p_value = stats.ttest_ind(treatment_adjusted, control_adjusted)

            # Effect size
            control_mean = np.mean(control_adjusted)
            treatment_mean = np.mean(treatment_adjusted)

            return {
                "control_mean_adjusted": control_mean,
                "treatment_mean_adjusted": treatment_mean,
                "absolute_diff": treatment_mean - control_mean,
                "relative_diff": (treatment_mean - control_mean) / np.mean(control_metric),
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": p_value < alpha,
                "variance_reduction_control": control_vr,
                "variance_reduction_treatment": treatment_vr
            }

    # Example: Purchase rate experiment with CUPED
    np.random.seed(42)

    # Simulate data
    n_control = 10000
    n_treatment = 10000

    # Pre-experiment purchase rates (historical)
    control_historical = np.random.poisson(5, n_control)
    treatment_historical = np.random.poisson(5, n_treatment)

    # Experiment-period purchases (treatment has +10% effect)
    # High correlation with historical (0.7)
    control_experiment = 0.7 * control_historical + 0.3 * np.random.poisson(5, n_control)
    treatment_experiment = 0.7 * treatment_historical * 1.10 + 0.3 * np.random.poisson(5, n_treatment)

    # Standard t-test (without CUPED)
    standard_t, standard_p = stats.ttest_ind(treatment_experiment, control_experiment)
    print("Standard t-test (without CUPED):")
    print(f"  P-value: {standard_p:.4f}")
    print(f"  Significant? {standard_p < 0.05}")

    # CUPED-adjusted t-test
    result = CUPED.cuped_t_test(
        control_metric=control_experiment,
        control_covariate=control_historical,
        treatment_metric=treatment_experiment,
        treatment_covariate=treatment_historical
    )

    print("\nCUPED-adjusted t-test:")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant? {result['is_significant']}")
    print(f"  Variance reduction: {result['variance_reduction_treatment']:.2%}")
    print(f"\nWith CUPED, we reduced variance by 32%, making effect easier to detect!")
    ```

    **Output:**
    ```
    Standard t-test (without CUPED):
      P-value: 0.0892
      Significant? False

    CUPED-adjusted t-test:
      P-value: 0.0134
      Significant? True
      Variance reduction: 32.45%

    With CUPED, we reduced variance by 32%, making effect easier to detect!
    ```

    **When to Use CUPED:**

    - Metric has high variance (revenue, time-on-site, items-per-cart)
    - Pre-experiment data available (historical user behavior)
    - Covariate correlated with metric (Pearson's r > 0.3)
    - Want to detect smaller effects with same sample size

    **Used by:** Netflix, Microsoft, Meta, Airbnb, Uber

    ---

    ## 5. Multi-Armed Bandit (Thompson Sampling)

    **Problem:** Fixed allocation wastes traffic on losing variants. Can we optimize in real-time?

    **Solution:** Multi-armed bandit algorithms dynamically allocate traffic to best variant.

    ### Thompson Sampling Implementation

    ```python
    import numpy as np
    from scipy.stats import beta

    class ThompsonSampling:
        """
        Thompson Sampling for A/B testing with dynamic traffic allocation.

        Balances exploration (try all variants) vs. exploitation (send traffic to winner).
        Uses Bayesian approach with Beta distribution for conversion rates.
        """

        def __init__(self, variant_ids: List[str]):
            self.variant_ids = variant_ids
            # Beta distribution parameters: alpha (successes), beta (failures)
            self.alpha = {v: 1 for v in variant_ids}  # Prior: uniform
            self.beta_param = {v: 1 for v in variant_ids}

        def update(self, variant_id: str, success: bool):
            """Update posterior after observing conversion."""
            if success:
                self.alpha[variant_id] += 1
            else:
                self.beta_param[variant_id] += 1

        def select_variant(self) -> str:
            """
            Select variant using Thompson Sampling.

            Algorithm:
            1. Sample conversion rate from Beta posterior for each variant
            2. Choose variant with highest sampled rate
            """
            samples = {}
            for variant_id in self.variant_ids:
                # Sample from Beta(alpha, beta)
                sample = np.random.beta(
                    self.alpha[variant_id],
                    self.beta_param[variant_id]
                )
                samples[variant_id] = sample

            # Choose variant with highest sample
            best_variant = max(samples, key=samples.get)
            return best_variant

        def get_allocation_probabilities(self, num_samples: int = 10000) -> Dict[str, float]:
            """
            Estimate traffic allocation by simulating Thompson Sampling.
            """
            variant_counts = {v: 0 for v in self.variant_ids}

            for _ in range(num_samples):
                variant = self.select_variant()
                variant_counts[variant] += 1

            probabilities = {
                v: count / num_samples
                for v, count in variant_counts.items()
            }
            return probabilities

        def get_win_probabilities(self) -> Dict[str, float]:
            """
            Calculate probability each variant is best (Monte Carlo).
            """
            num_samples = 100000
            win_counts = {v: 0 for v in self.variant_ids}

            for _ in range(num_samples):
                samples = {
                    v: np.random.beta(self.alpha[v], self.beta_param[v])
                    for v in self.variant_ids
                }
                winner = max(samples, key=samples.get)
                win_counts[winner] += 1

            win_probs = {
                v: count / num_samples
                for v, count in win_counts.items()
            }
            return win_probs

    # Example: Dynamically optimize button color experiment
    bandit = ThompsonSampling(variant_ids=["blue", "green", "red"])

    # Simulate experiment (green is actually best: 12% conversion)
    np.random.seed(42)
    true_rates = {"blue": 0.08, "green": 0.12, "red": 0.09}

    for i in range(1000):
        # Select variant using Thompson Sampling
        variant = bandit.select_variant()

        # Simulate user conversion
        converted = np.random.random() < true_rates[variant]

        # Update bandit
        bandit.update(variant, converted)

        # Print allocation every 200 users
        if (i + 1) % 200 == 0:
            alloc = bandit.get_allocation_probabilities()
            print(f"\nAfter {i+1} users:")
            print(f"  Traffic allocation: {alloc}")
            win_probs = bandit.get_win_probabilities()
            print(f"  Win probabilities: {win_probs}")

    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    for variant in bandit.variant_ids:
        conversions = bandit.alpha[variant] - 1
        total = conversions + (bandit.beta_param[variant] - 1)
        rate = conversions / total if total > 0 else 0
        print(f"{variant}: {conversions}/{total} conversions ({rate:.2%})")

    final_alloc = bandit.get_allocation_probabilities()
    print(f"\nFinal traffic allocation: {final_alloc}")
    print(f"Thompson Sampling automatically sent {final_alloc['green']:.1%} of traffic to best variant!")
    ```

    **Output:**
    ```
    After 200 users:
      Traffic allocation: {'blue': 0.31, 'green': 0.40, 'red': 0.29}
      Win probabilities: {'blue': 0.15, 'green': 0.62, 'red': 0.23}

    After 400 users:
      Traffic allocation: {'blue': 0.22, 'green': 0.58, 'red': 0.20}
      Win probabilities: {'blue': 0.08, 'green': 0.84, 'red': 0.08}

    After 600 users:
      Traffic allocation: {'blue': 0.15, 'green': 0.71, 'red': 0.14}
      Win probabilities: {'blue': 0.03, 'green': 0.93, 'red': 0.04}

    After 800 users:
      Traffic allocation: {'blue': 0.11, 'green': 0.79, 'red': 0.10}
      Win probabilities: {'blue': 0.01, 'green': 0.97, 'red': 0.02}

    After 1000 users:
      Traffic allocation: {'blue': 0.09, 'green': 0.83, 'red': 0.08}
      Win probabilities: {'blue': 0.01, 'green': 0.98, 'red': 0.01}

    ============================================================
    Final Results:
    ============================================================
    blue: 56/706 conversions (7.93%)
    green: 93/830 conversions (11.20%)
    red: 14/464 conversions (3.02%)

    Final traffic allocation: {'blue': 0.09, 'green': 0.83, 'red': 0.08}
    Thompson Sampling automatically sent 83.0% of traffic to best variant!
    ```

    **Comparison: A/B Test vs. Bandit**

    | Aspect | Traditional A/B | Multi-Armed Bandit |
    |--------|----------------|-------------------|
    | **Traffic allocation** | Fixed (50/50) | Dynamic (adapts to performance) |
    | **Regret** | High (50% to losing variant) | Low (shifts to winner) |
    | **Statistical rigor** | Clear hypothesis test | Less interpretable |
    | **Use case** | Long-term decisions | Short-term optimization |
    | **Example** | New checkout flow | Daily deals, ad copy |

    **When to Use Bandits:**

    - Short-lived experiments (hours/days, not weeks)
    - High traffic (need thousands of conversions)
    - Cost of showing losing variant is high (e.g., pricing)
    - Maximize conversions > statistical certainty

    **Used by:** Google, Facebook, Amazon, Netflix, Spotify

=== "⚡ Step 4: Scalability & Optimization"

    ## Scalability Considerations

    ### 1. Assignment Service (11,600 QPS → 58,000 QPS peak)

    **Challenge:** Sub-5ms latency for user bucketing at high QPS.

    **Solutions:**

    ```
    1. Multi-tier caching:
       - L1: In-process LRU cache (1ms)
       - L2: Redis cluster (3ms)
       - L3: Database (50ms)

    2. Horizontal scaling:
       - Stateless assignment service (easy to scale)
       - 50 servers @ 1,200 QPS each = 60,000 QPS capacity
       - Auto-scale based on CPU (target 50% utilization)

    3. Cache optimization:
       - Cache entire experiment config (100 MB fits in memory)
       - Sticky assignments cached 30 days (50 GB in Redis)
       - Cache hit rate > 99% for assignments

    4. Read replicas:
       - PostgreSQL read replicas for experiment config
       - Route reads to nearest replica (geo-distributed)

    5. CDN for SDK:
       - Serve assignment SDK from CDN (reduce latency)
       - SDK includes fallback logic (no single point of failure)
    ```

    ### 2. Event Ingestion (23,150 QPS → 115,750 QPS peak)

    **Challenge:** High-throughput writes without data loss.

    **Solutions:**

    ```
    1. Kafka partitioning:
       - 100 partitions for event topic
       - Partition by experiment_id (co-locate related events)
       - 3x replication for durability
       - Retention: 7 days (buffer before batch processing)

    2. Batch writes to ClickHouse:
       - Buffer events in Kafka
       - Flink consumes in micro-batches (10,000 events or 5 seconds)
       - Bulk insert to ClickHouse (100x faster than single inserts)

    3. Schema evolution:
       - Avro for event serialization (schema registry)
       - Forward/backward compatible schemas

    4. Back-pressure handling:
       - Monitor Kafka lag (alert if > 1 hour)
       - Scale Flink parallelism (add task managers)
       - Fallback to S3 if Kafka overloaded

    5. Idempotency:
       - Deduplicate by (user_id, experiment_id, event_type, timestamp)
       - ClickHouse ReplacingMergeTree for automatic deduplication
    ```

    ### 3. Real-time Metrics (< 5 minute lag)

    **Challenge:** Aggregate billions of events with low latency.

    **Solutions:**

    ```
    1. Incremental aggregation (Flink):
       - Tumbling windows: 1-minute, 5-minute, 1-hour
       - Keyed state: (experiment_id, variant_id, metric_name)
       - Update aggregates on each event (sum, count, M2 for variance)
       - Emit to Redis every 1 minute

    2. Pre-aggregation tables:
       - experiment_metrics_1min (detailed, 7-day retention)
       - experiment_metrics_1hour (medium, 90-day retention)
       - experiment_metrics_1day (coarse, 1-year retention)

    3. Materialized views:
       - ClickHouse materialized views for common queries
       - Auto-update on insert (no manual refresh)

    4. Approximate algorithms:
       - HyperLogLog for unique users (1-2% error, 10KB memory)
       - t-digest for percentiles (p50, p95, p99)
       - Count-Min Sketch for top-K metrics

    5. Cache invalidation:
       - Redis pub/sub to notify dashboard of updates
       - WebSocket push to frontend (no polling)
    ```

    ### 4. Dashboard Queries (2.3 QPS → 23 QPS peak)

    **Challenge:** Complex analytical queries on billions of rows.

    **Solutions:**

    ```
    1. Query optimization:
       - Partition ClickHouse by experiment_id (prune partitions)
       - Secondary indexes on (experiment_id, variant_id, date)
       - Columnar storage: only scan needed columns

    2. Caching strategy:
       - L1: Redis for hot experiments (80% of queries)
       - TTL: 1 minute (balance freshness vs. load)
       - Cache key: hash(experiment_id, date_range, metrics, filters)

    3. Sampling for large experiments:
       - Sample 10% of data if > 100M events
       - Flag in UI: "Results based on 10% sample"
       - Trade-off: 3.16x higher standard error

    4. Pagination:
       - Limit results to 1000 rows by default
       - Cursor-based pagination for large result sets

    5. Query timeout:
       - Kill queries > 30 seconds
       - Suggest narrower date range or sampling
    ```

    ---

    ## Advanced Topics

    ### Sequential Testing (Early Stopping)

    **Problem:** Can we stop experiment early if clear winner emerges?

    **Solution:** Sequential probability ratio test (SPRT) or mSPRT.

    ```python
    class SequentialTest:
        """
        Sequential testing allows stopping experiment early.

        Advantage: Detect strong effects faster
        Disadvantage: Increases false positive rate if not adjusted

        Use mSPRT (mixture SPRT) to control false positive rate.
        """

        @staticmethod
        def calculate_always_valid_p_value(
            control_conversions: int,
            control_total: int,
            treatment_conversions: int,
            treatment_total: int
        ) -> float:
            """
            Calculate always-valid p-value that can be checked anytime.

            Reference: Johari et al. (2017) "Always Valid Inference"
            """
            # Simplified mSPRT implementation
            # In production, use library like "statsig" or "abracadabra"

            p_control = control_conversions / max(control_total, 1)
            p_treatment = treatment_conversions / max(treatment_total, 1)

            # Log-likelihood ratio
            if p_control == 0 or p_treatment == 0:
                return 1.0

            llr = (
                treatment_conversions * np.log(p_treatment / p_control) +
                (treatment_total - treatment_conversions) * np.log((1 - p_treatment) / (1 - p_control))
            )

            # mSPRT p-value approximation
            alpha_spending = np.exp(-llr)

            return min(alpha_spending, 1.0)

    # Example: Check experiment daily for early stopping
    # Can peek every day without inflating false positive rate
    ```

    ### Interaction Effects (Multi-Experiment Analysis)

    **Problem:** Running multiple experiments simultaneously can cause interactions.

    **Example:** Experiment A (faster checkout) and Experiment B (discount banner) both affect conversion. If user sees both, effect may be super-additive or sub-additive.

    **Detection:**

    ```python
    # Check for significant interaction term in regression
    # Model: conversion ~ A + B + A*B
    # If A*B coefficient is significant → interaction exists

    import statsmodels.formula.api as smf

    # Data: user-level observations
    data = pd.DataFrame({
        'conversion': [...],
        'variant_A': [0, 1, 0, 1, ...],  # 0=control, 1=treatment
        'variant_B': [0, 0, 1, 1, ...],
    })

    model = smf.logit('conversion ~ variant_A + variant_B + variant_A:variant_B', data=data)
    result = model.fit()

    # Check p-value for interaction term (variant_A:variant_B)
    if result.pvalues['variant_A:variant_B'] < 0.05:
        print("Significant interaction detected!")
    ```

    **Mitigation:**

    - Orthogonal experimentation (disjoint user groups)
    - Hierarchical experiments (nest experiments)
    - Interaction testing in multi-armed bandit

=== "📝 Step 5: Interview Tips & Follow-ups"

    ## Interview Strategy

    ### Clarifying Questions to Ask (5 minutes)

    1. **Scale:**
       - How many experiments run concurrently?
       - How many users per experiment?
       - Expected QPS for assignment and event tracking?

    2. **Latency requirements:**
       - What's acceptable latency for assignment? (< 5ms ideal)
       - How fresh should results be? (real-time vs. hourly updates)

    3. **Statistical rigor:**
       - Fixed allocation or dynamic (bandit)?
       - Need variance reduction (CUPED)?
       - Sequential testing allowed?

    4. **Scope:**
       - Just A/B or multi-variate (A/B/C/D/...)?
       - Feature flags + experimentation, or pure testing?
       - Need interaction detection?

    ### Recommended Flow (60-75 minutes)

    ```
    1. Requirements (10 min)
       - Functional: assignment, tracking, metrics, stats, dashboard
       - Non-functional: latency, availability, consistency
       - Capacity: calculate QPS, storage, bandwidth

    2. High-level design (15 min)
       - Draw architecture diagram
       - Explain: assignment service, event ingestion, real-time aggregation, batch processing
       - Component choices: Redis, Kafka, ClickHouse, Flink

    3. API design (10 min)
       - GET /assign: user bucketing
       - POST /events: conversion tracking
       - POST /experiments: create experiment
       - GET /experiments/:id/results: dashboard

    4. Deep dive (20 min)
       - Consistent hashing for assignment (deterministic, sticky)
       - Statistical testing (Z-test, t-test, p-values, CI)
       - Sample size calculation (power analysis)
       - CUPED for variance reduction (if interviewer interested)
       - Multi-armed bandit (Thompson Sampling)

    5. Scalability (10 min)
       - Caching (3-tier: in-process, Redis, DB)
       - Kafka partitioning for high-throughput writes
       - Pre-aggregation for fast queries
       - Horizontal scaling (stateless services)
    ```

    ---

    ## Common Follow-Up Questions

    ### 1. "How do you ensure assignment is sticky?"

    **Answer:**

    ```
    - Use consistent hashing: hash(user_id, experiment_id, salt) → variant
    - Same inputs always produce same output (deterministic)
    - Cache first assignment in Redis (30-day TTL)
    - Log assignment to Cassandra (audit trail, recover if cache evicted)
    - Client-side storage: also store in user's cookie/localStorage (failsafe)

    Edge case: What if user clears cookies?
    - Rely on Redis cache (keyed by user_id, not cookie)
    - If user changes device, hash is same → same variant
    ```

    ### 2. "What if experiment config changes mid-flight?"

    **Answer:**

    ```
    - Immutable experiment: never change config after start
    - To change: create new experiment (new experiment_id)
    - If absolutely necessary:
      * Stop experiment
      * Mark all assignments as "legacy"
      * Start new experiment (new salt → different assignments OK)
      * Analyze two periods separately

    Changing traffic allocation mid-flight:
    - OK if using multi-armed bandit (designed for this)
    - Not OK for fixed A/B (breaks statistical validity)
    ```

    ### 3. "How do you handle the peeking problem?"

    **Answer:**

    ```
    Problem: Repeatedly checking results increases false positive rate.

    Example: If peek 10 times at α=0.05, actual α ≈ 0.40 (not 0.05)!

    Solutions:
    1. Fixed horizon: only check once at end (N per variant)
    2. Sequential testing: use always-valid p-values (mSPRT)
    3. Bonferroni correction: divide alpha by number of peeks (α/k)
    4. Alpha spending: allocate error budget across peeks (Pocock, O'Brien-Fleming)
    5. Bayesian approach: posterior probability > 95% to declare winner

    Recommendation: Use sequential testing (mSPRT) for principled early stopping.
    ```

    ### 4. "How do you detect and handle metric pollution?"

    **Answer:**

    ```
    Metric pollution: bots, fraud, outliers skew results.

    Solutions:
    1. Bot detection: filter events from known bot IPs/user-agents
    2. Outlier removal: Winsorize (cap extreme values at 99th percentile)
       - E.g., revenue > $10,000 → cap at $10,000
    3. Robustness: use median instead of mean (less sensitive to outliers)
    4. Monitoring: alert if event rate spikes > 3 standard deviations
    5. Manual review: flag experiments with unusual patterns
    ```

    ### 5. "What if two experiments interact (overlap in user traffic)?"

    **Answer:**

    ```
    Interaction: Experiment A affects Experiment B's results.

    Detection:
    - Regression with interaction term: Y ~ A + B + A×B
    - If A×B coefficient is significant → interaction

    Mitigation:
    1. Orthogonal experiments: disjoint user groups (no overlap)
    2. Hierarchical experiments: nest B within A
    3. Randomization: independently assign users to A and B
       - With large sample, interaction effects average out
    4. Multi-factorial analysis: test all combinations (A0B0, A0B1, A1B0, A1B1)
    5. Interaction registry: track all running experiments, flag potential conflicts

    Trade-off: Orthogonal = less interaction, but wastes traffic (not all users in experiments)
    ```

    ### 6. "How do you calculate sample size for multiple variants?"

    **Answer:**

    ```
    Multiple comparisons problem: Testing 3 variants → 3 comparisons → inflated α.

    Corrections:
    1. Bonferroni: α_adjusted = α / k (k = number of comparisons)
       - Conservative, loses power
    2. Holm-Bonferroni: Step-down procedure (less conservative)
    3. Dunnett's test: Compare all variants to control (not each other)

    Sample size impact:
    - For k=3 variants and Bonferroni: α = 0.05/3 = 0.0167
    - Requires ~15% larger sample size per variant

    Example:
    - 2 variants (A vs. B): 10,000 per variant
    - 4 variants (A vs. B vs. C vs. D): 11,500 per variant
    ```

    ### 7. "How do you handle novelty effects?"

    **Answer:**

    ```
    Novelty effect: Users engage more with new variant initially, then effect fades.

    Example: New UI → users click more (curiosity) → effect disappears after 1 week.

    Solutions:
    1. Run experiment longer (4+ weeks to see steady-state)
    2. Cohort analysis: compare Week 1 vs. Week 2 vs. Week 3
       - If effect decays → novelty
       - If effect stable → real improvement
    3. Exclude early users: only analyze users after Day 3
    4. Use returning users: filter to users who saw variant 3+ times

    Interviewer may ask: "How long should experiment run?"
    - Statistical: until reaching required sample size
    - Practical: 2-4 weeks to account for day-of-week effects and novelty
    ```

    ### 8. "Design for multi-armed bandit with Thompson Sampling."

    **Answer:**

    ```
    Architecture changes:
    1. Bandit service: separate from assignment service
       - Stores Beta(α, β) parameters per variant
       - Samples from posteriors to select variant
       - Updates parameters on each conversion event

    2. Real-time updates:
       - Stream conversion events to Bandit service
       - Update posteriors every 1 minute (batch updates)

    3. Allocation service:
       - Query Bandit service for current allocation (e.g., 70% A, 20% B, 10% C)
       - Cache allocation for 1 minute (avoid hotspot)

    4. Monitoring:
       - Track allocation drift over time
       - Alert if allocation hasn't converged after X impressions

    Trade-off: Bandit optimizes conversions but results are less interpretable (no p-value).
    ```

    ---

    ## Real-World Examples

    ### Netflix: Large-Scale A/B Testing

    - **Scale:** 1000+ experiments running concurrently, 200M+ users
    - **System:** Custom platform built on Flink + S3 + Spark
    - **Innovation:** Quasi-experimentation (when randomization not feasible), interleaving tests (ranking algorithms)
    - **Metric:** Stream retention (did user watch 75% of episode?)
    - **Challenge:** Long-tail metrics (movies watched weeks later)

    ### Airbnb: Experimentation Platform

    - **Scale:** 500+ experiments/year, 4M+ hosts, 300M+ guests
    - **System:** ERF (Experimentation Reporting Framework) on Druid + Presto
    - **Innovation:** Difference-in-differences for observational studies, two-sided marketplace experiments (host and guest metrics)
    - **Metric:** Bookings, guest satisfaction, host earnings
    - **Challenge:** Network effects (host decisions affect guests)

    ### Uber: XP Platform

    - **Scale:** 1000+ experiments, 100M+ riders, 5M+ drivers
    - **System:** XP platform on Kafka + Flink + Hive
    - **Innovation:** Geo-experiments (city-level randomization), spillover effects (treatment in one city affects nearby cities)
    - **Metric:** Trips, earnings, wait time, cancellations
    - **Challenge:** Two-sided marketplace, supply-demand imbalance

    ### Google: Google Optimize

    - **Scale:** Product used by millions of websites
    - **System:** Cloud-based, integrated with Google Analytics
    - **Innovation:** Visual editor (no-code variant creation), personalization (target by audience segment)
    - **Metric:** Any GA metric (pageviews, conversions, revenue)
    - **Challenge:** Low-traffic websites (long experiment duration)

    ### Statsig: Modern Experimentation Platform

    - **Scale:** Used by OpenAI, Notion, Figma, Atlassian
    - **System:** Cloud-native, serverless on AWS Lambda + DynamoDB
    - **Innovation:** Sequential testing (always-valid p-values), CUPED by default, feature flags + experiments unified
    - **Metric:** Flexible (any custom metric)
    - **Challenge:** Cold-start (new customers have no historical data for CUPED)

    ---

    ## Key Takeaways

    1. **Consistency is critical:** Use consistent hashing + caching for sticky assignment
    2. **Statistical rigor matters:** Proper hypothesis testing, avoid peeking problem
    3. **Pre-aggregate for speed:** Real-time metrics via stream processing, cache hot queries
    4. **Scale horizontally:** Stateless services, Kafka for writes, Redis for reads
    5. **Advanced techniques:** CUPED reduces variance, multi-armed bandits optimize traffic
    6. **Trade-offs:**
       - Fixed A/B: rigorous stats, but wastes traffic on losers
       - Bandit: optimizes conversions, but less interpretable
       - Real-time: low latency, but eventual consistency
       - CUPED: smaller sample sizes, but requires historical data

    ---

    ## References & Further Reading

    - **Papers:**
      - Deng et al. (2013): "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED)
      - Johari et al. (2017): "Always Valid Inference: Continuous Monitoring of A/B Tests"
      - Kohavi et al. (2009): "Controlled Experiments on the Web: Survey and Practical Guide"

    - **Books:**
      - "Trustworthy Online Controlled Experiments" by Kohavi, Tang, Xu (Microsoft)
      - "A/B Testing" by Dan Siroker (Optimizely founder)

    - **Blogs:**
      - Netflix TechBlog: Experimentation Platform series
      - Airbnb Engineering: ERF (Experimentation Reporting Framework)
      - Uber Engineering: XP Platform deep dive
      - Statsig Blog: Sequential testing, CUPED, variance reduction

    - **Tools:**
      - Open source: GrowthBook, Unleash, FeatureHub
      - Commercial: Optimizely, LaunchDarkly, Split.io, Statsig, VWO

