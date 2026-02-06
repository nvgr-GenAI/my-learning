# Design an Industrial IoT Monitoring System (Predictive Maintenance)

A comprehensive Industrial IoT platform that enables real-time monitoring of manufacturing equipment, predictive maintenance using machine learning, digital twin synchronization, edge computing for low-latency analytics, and automated maintenance scheduling to minimize downtime and optimize operational efficiency.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 75-90 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100K machines, 10M sensors, 1B data points/day, 50K prediction models, 10K maintenance events/day |
| **Key Challenges** | High-frequency sensor data ingestion, edge computing for real-time anomaly detection, predictive maintenance with ML, digital twin state synchronization, OPC UA protocol integration, remaining useful life (RUL) prediction, multi-site coordination |
| **Core Concepts** | OPC UA server, edge gateways, time-series analytics, survival analysis, digital twin, anomaly detection, predictive maintenance, edge ML inference, industrial protocols, SCADA integration |
| **Companies** | GE Predix, Siemens MindSphere, AWS IoT, Honeywell Connected Plant, PTC ThingWorx, Microsoft Azure IoT |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Sensor Data Collection** | Real-time collection from 10M sensors via OPC UA, Modbus, MQTT | P0 (Must have) |
    | **Edge Analytics** | Real-time anomaly detection and filtering at edge gateways | P0 (Must have) |
    | **Digital Twin** | Virtual representation of physical assets with state synchronization | P0 (Must have) |
    | **Predictive Maintenance** | ML-based prediction of equipment failures and RUL | P0 (Must have) |
    | **Maintenance Scheduling** | Automated work order creation and technician assignment | P0 (Must have) |
    | **Alert Management** | Real-time alerts for anomalies, predicted failures, threshold violations | P0 (Must have) |
    | **Historical Analytics** | Time-series analysis, trend detection, performance metrics | P1 (Should have) |
    | **Asset Management** | Equipment inventory, specifications, maintenance history | P1 (Should have) |
    | **Multi-site Coordination** | Centralized monitoring across multiple factories | P1 (Should have) |
    | **Integration APIs** | Connect with ERP, CMMS, MES systems | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Supply chain management and inventory optimization
    - Production planning and scheduling (MES functions)
    - Quality control and inspection systems
    - Energy management and optimization
    - Safety and security monitoring (separate system)
    - Real-time control loops (PLC/DCS responsibility)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Manufacturing downtime is extremely costly ($250K/hour avg) |
    | **Edge Latency** | < 100ms for anomaly detection | Real-time detection prevents cascading failures |
    | **Cloud Latency** | < 2s for dashboards | Fast operator response to issues |
    | **Data Loss** | < 0.01% of sensor readings | Critical for accurate failure prediction |
    | **Prediction Accuracy** | > 85% for failure prediction | Balance false positives vs. missed failures |
    | **RUL Prediction** | ± 10% accuracy | Enable proactive maintenance scheduling |
    | **Data Retention** | 5 years full data | Regulatory compliance and long-term analysis |
    | **Scalability** | 100K+ machines, 10M+ sensors | Support large manufacturing operations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Manufacturing Environment:
    - Total factories: 50 sites
    - Machines per factory: 2,000 machines
    - Total machines: 100,000 machines
    - Sensors per machine: 100 sensors (temp, vibration, pressure, flow, etc.)
    - Total sensors: 10,000,000 sensors

    Sensor Data Collection:
    - High-frequency sensors (vibration, pressure): 1 Hz (1 reading/sec)
      - Count: 2M sensors
      - Readings/day: 2M × 86,400 = 172.8B/day

    - Medium-frequency sensors (temperature, flow): 0.1 Hz (1 reading/10 sec)
      - Count: 5M sensors
      - Readings/day: 5M × 8,640 = 43.2B/day

    - Low-frequency sensors (position, status): 0.01 Hz (1 reading/100 sec)
      - Count: 3M sensors
      - Readings/day: 3M × 864 = 2.6B/day

    Total readings: 172.8B + 43.2B + 2.6B = 218.6B readings/day
    Average: 2.5M readings/sec
    Peak (manufacturing shift changes): 7.5M readings/sec

    Edge Processing:
    - Edge gateways: 5,000 gateways (20 machines per gateway)
    - Processing per gateway: 2,000 sensors × 0.5 Hz avg = 1,000 readings/sec
    - Edge anomaly detection: 1,000 models per gateway
    - Edge-to-cloud traffic: 10% of raw data (after filtering) = 250K readings/sec

    Predictive Maintenance:
    - ML models: 50K models (failure prediction per machine component)
    - Prediction frequency: Every 1 hour per model
    - Total predictions: 50K × 24 = 1.2M predictions/day = 14 predictions/sec
    - Model training: Weekly batch jobs (10K models/week)

    Digital Twin Updates:
    - Twin instances: 100K twins (one per machine)
    - State update frequency: 1 Hz (critical parameters)
    - Total updates: 100K × 86,400 = 8.6B updates/day = 100K updates/sec
    - Twin queries: 10K queries/sec (dashboards, APIs)

    Alerts and Maintenance:
    - Anomaly alerts: 50K alerts/day (0.5% anomaly rate)
    - Predicted failures: 5K/day (5% monthly failure rate)
    - Maintenance work orders: 10K/day (proactive + reactive)
    - Technician interactions: 50K queries/day
    ```

    ### Storage Estimates

    ```
    Per sensor reading (compressed):
    - Sensor ID: 8 bytes (UUID)
    - Timestamp: 8 bytes (Unix timestamp, millisecond precision)
    - Value: 4 bytes (float32)
    - Quality flag: 1 byte (good, bad, uncertain)
    - Total per reading: 21 bytes

    Raw sensor data (before edge filtering):
    - Daily: 218.6B readings × 21 bytes = 4.59 TB/day
    - Yearly: 4.59 TB × 365 = 1.68 PB/year
    - 5 years: 8.4 PB

    After edge filtering (90% reduction):
    - Daily: 4.59 TB × 0.1 = 459 GB/day
    - Yearly: 167.5 TB/year
    - 5 years: 837.5 TB

    With time-series compression (10:1 ratio):
    - 5 years compressed: 837.5 TB / 10 = 83.75 TB

    Multi-tier retention strategy:

    Tier 1: Full resolution - 90 days
    - Storage: 459 GB/day × 90 = 41.3 TB
    - Compressed: 4.13 TB
    - Use: Real-time monitoring, immediate analysis

    Tier 2: 1-minute aggregation - 1 year
    - Reduction: 60x fewer points (for 1 Hz sensors)
    - Storage: 167.5 TB / 60 = 2.79 TB
    - Compressed: 279 GB
    - Use: Historical trends, model training

    Tier 3: 1-hour aggregation - 5 years
    - Reduction: 3600x fewer points
    - Storage: 837.5 TB × 5 / 3600 = 1.16 TB
    - Compressed: 116 GB
    - Use: Long-term trends, compliance

    Total time-series storage: 4.13 TB + 279 GB + 116 GB = 4.52 TB
    With replication (3x): 13.6 TB

    ML Models and Features:
    - 50K models × 100 MB (model + features) = 5 TB
    - Historical feature store: 20 TB
    - Total ML storage: 25 TB

    Asset and Maintenance Data:
    - 100K machines × 10 KB (metadata, specs) = 1 GB
    - 10M maintenance records × 5 KB = 50 GB
    - Digital twin state: 100K twins × 1 MB = 100 GB
    - Alerts history: 50K/day × 2 KB × 365 = 36.5 GB/year
    - Total operational data: 200 GB

    Total Storage: 13.6 TB (time-series) + 25 TB (ML) + 200 GB (ops) = 38.8 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (sensor data to edge):
    - 2.5M readings/sec × 21 bytes = 52.5 MB/sec = 420 Mbps per factory
    - 50 factories: 21 Gbps total
    - Protocol overhead (OPC UA, Modbus TCP): 1.5x = 31.5 Gbps

    Edge to Cloud:
    - After filtering: 250K readings/sec × 21 bytes = 5.25 MB/sec = 42 Mbps
    - Compressed (2:1): 21 Mbps
    - With protocol overhead: 31.5 Mbps
    - All factories: 1.58 Gbps

    Egress (dashboards, APIs, analytics):
    - Real-time dashboards: 5K operators × 100 KB/sec = 500 MB/sec
    - Historical queries: 1K queries/sec × 1 MB = 1 GB/sec
    - ML model distribution: 10K models × 100 MB / 3600 sec = 278 MB/sec
    - Total egress: 1.78 GB/sec = 14.2 Gbps

    Total bandwidth: 1.58 Gbps (ingress) + 14.2 Gbps (egress) = 15.8 Gbps
    ```

    ### Server Estimates

    ```
    Edge Gateways:
    - Total gateways: 5,000 gateways (20 machines per gateway)
    - CPU: 8 cores per gateway (real-time processing)
    - Memory: 16 GB per gateway (buffering, ML models)
    - Storage: 500 GB local SSD (24-hour buffer)

    Data Ingestion Layer (Cloud):
    - 250K readings/sec / 5,000 per node = 50 ingestion nodes
    - CPU: 4 cores per node
    - Memory: 16 GB per node

    Time-Series Database:
    - 13.6 TB total / 2 TB per node = 7 TSDB nodes
    - CPU: 32 cores per node (compression, indexing)
    - Memory: 256 GB per node (hot data)
    - Disk: 2 TB NVMe SSD per node

    ML Training Cluster:
    - 50K models, weekly retraining
    - GPU nodes: 20 nodes × 8 GPUs = 160 GPUs
    - CPU: 64 cores per node
    - Memory: 512 GB per node
    - Training time: 10K models/week / 20 nodes = 500 models/node/week

    ML Inference (Prediction) Layer:
    - 14 predictions/sec / 10 per node = 2 nodes (with redundancy: 4 nodes)
    - CPU: 16 cores per node
    - Memory: 64 GB per node (model cache)

    Digital Twin Service:
    - 100K updates/sec / 10K per node = 10 twin nodes
    - CPU: 16 cores per node
    - Memory: 32 GB per node (twin state cache)

    Query/API Layer:
    - 10K queries/sec / 200 per node = 50 API nodes
    - CPU: 8 cores per node
    - Memory: 32 GB per node

    Total Infrastructure:
    - Edge: 5,000 gateways
    - Cloud: 50 (ingestion) + 7 (TSDB) + 20 (ML training) + 4 (ML inference) + 10 (twin) + 50 (API) = 141 nodes
    ```

    ---

    ## Key Assumptions

    1. 100 sensors per machine (mix of high, medium, low frequency)
    2. Edge filtering reduces data by 90% (only anomalies sent to cloud)
    3. 10:1 compression ratio for time-series data
    4. 85% prediction accuracy for failure detection
    5. 5% monthly failure rate across all equipment
    6. OPC UA protocol for majority of data collection
    7. 24-hour edge buffer capacity for network outages
    8. Weekly model retraining cycle

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Edge-first architecture** - Process data at edge, send only insights to cloud
    2. **Digital twin pattern** - Virtual representation synchronized with physical assets
    3. **ML-driven maintenance** - Predictive models for failure forecasting
    4. **Industrial protocol support** - OPC UA, Modbus, MQTT, REST
    5. **Multi-tier storage** - Balance latency, cost, and retention requirements

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Factory Floor (Edge)"
            subgraph "Production Lines"
                PLC1[PLC/DCS<br/>Line Control]
                PLC2[PLC/DCS<br/>Line Control]
                Sensors1[Sensors<br/>Temp, Vibration<br/>Pressure, Flow]
                Sensors2[Sensors<br/>Position, Speed<br/>Torque, Current]
                SCADA[SCADA System<br/>Local HMI]
            end

            subgraph "Edge Gateway 1"
                OPC_Server1[OPC UA Server<br/>Protocol bridge]
                EdgeProcessor1[Edge Processor<br/>Real-time analytics]
                EdgeML1[Edge ML Engine<br/>Anomaly detection]
                EdgeBuffer1[Local Buffer<br/>24h cache]
            end

            subgraph "Edge Gateway 2"
                OPC_Server2[OPC UA Server<br/>Protocol bridge]
                EdgeProcessor2[Edge Processor<br/>Real-time analytics]
                EdgeML2[Edge ML Engine<br/>Anomaly detection]
                EdgeBuffer2[Local Buffer<br/>24h cache]
            end
        end

        subgraph "Cloud Platform"
            subgraph "Data Ingestion"
                Gateway[API Gateway<br/>Auth, rate limiting]
                MsgBroker[Message Broker<br/>Kafka<br/>Data streams]
                Validator[Data Validator<br/>Quality checks]
            end

            subgraph "Time-Series Storage"
                TSDB_Hot[(Hot Storage<br/>TimescaleDB<br/>90 days<br/>Full resolution)]
                TSDB_Warm[(Warm Storage<br/>TimescaleDB<br/>1 year<br/>1-min aggregation)]
                TSDB_Cold[(Cold Storage<br/>S3/Parquet<br/>5 years<br/>1-hour aggregation)]
            end

            subgraph "Digital Twin Engine"
                TwinRegistry[Twin Registry<br/>Asset definitions]
                TwinState[Twin State Store<br/>Real-time state]
                TwinSync[State Synchronizer<br/>Edge-cloud sync]
                TwinQuery[Twin Query API<br/>GraphQL]
            end

            subgraph "ML & Analytics"
                FeatureStore[Feature Store<br/>ML features<br/>Time windows]
                MLTraining[ML Training<br/>Survival analysis<br/>RUL prediction]
                MLInference[ML Inference<br/>Real-time scoring]
                AnomalyDetector[Anomaly Detector<br/>Pattern detection]
            end

            subgraph "Maintenance Management"
                MaintenanceScheduler[Maintenance Scheduler<br/>Work order creation]
                RulEngine[RUL Engine<br/>Remaining useful life]
                AlertManager[Alert Manager<br/>Rules engine]
                WorkOrderService[Work Order Service<br/>CMMS integration]
            end

            subgraph "Application Layer"
                API[API Layer<br/>REST/GraphQL]
                Dashboard[Operations Dashboard<br/>Real-time monitoring]
                MaintenanceApp[Maintenance App<br/>Mobile + Web]
                AnalyticsPortal[Analytics Portal<br/>BI, reports]
            end

            subgraph "External Systems"
                ERP[ERP System<br/>SAP, Oracle]
                CMMS[CMMS<br/>Maximo, SAP PM]
                MES[MES System<br/>Production data]
            end
        end

        Sensors1 -->|Modbus TCP| OPC_Server1
        Sensors2 -->|OPC UA| OPC_Server1
        PLC1 --> OPC_Server1
        PLC2 --> OPC_Server2
        SCADA -.->|Read-only| OPC_Server1

        OPC_Server1 --> EdgeProcessor1
        EdgeProcessor1 --> EdgeML1
        EdgeML1 --> EdgeBuffer1

        OPC_Server2 --> EdgeProcessor2
        EdgeProcessor2 --> EdgeML2
        EdgeML2 --> EdgeBuffer2

        EdgeBuffer1 -->|MQTT/HTTPS| Gateway
        EdgeBuffer2 -->|MQTT/HTTPS| Gateway

        Gateway --> MsgBroker
        MsgBroker --> Validator
        Validator --> TSDB_Hot

        TSDB_Hot --> TSDB_Warm
        TSDB_Warm --> TSDB_Cold

        MsgBroker --> TwinSync
        TwinSync --> TwinState
        TwinRegistry --> TwinState
        TwinState --> TwinQuery

        TSDB_Hot --> FeatureStore
        FeatureStore --> MLTraining
        MLTraining -.->|Weekly| MLInference
        MLInference --> RulEngine

        TSDB_Hot --> AnomalyDetector
        AnomalyDetector --> AlertManager
        RulEngine --> AlertManager

        AlertManager --> MaintenanceScheduler
        MaintenanceScheduler --> WorkOrderService
        WorkOrderService --> CMMS

        API --> Dashboard
        API --> MaintenanceApp
        API --> AnalyticsPortal

        TwinQuery --> API
        TSDB_Hot --> API
        MLInference --> API

        WorkOrderService --> ERP
        FeatureStore --> MES

        MLTraining -.->|Deploy models| EdgeML1
        MLTraining -.->|Deploy models| EdgeML2

        style EdgeML1 fill:#e8f5e9
        style EdgeML2 fill:#e8f5e9
        style TSDB_Hot fill:#ffe1e1
        style TSDB_Warm fill:#fff4e1
        style TSDB_Cold fill:#f0f0f0
        style TwinState fill:#e1f5ff
        style MLInference fill:#f3e5f5
        style AlertManager fill:#fff9c4
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **OPC UA Server** | Standard industrial protocol for sensor data collection, secure, vendor-neutral | Modbus only (limited features), proprietary protocols (vendor lock-in) |
    | **Edge Gateways** | Real-time processing, reduce cloud bandwidth by 90%, operate during network outages | Cloud-only processing (high latency, bandwidth costs), direct PLC connections (security risk) |
    | **Edge ML** | < 100ms anomaly detection, no cloud dependency for critical alerts | Cloud-only ML (400ms+ latency), rule-based only (low accuracy) |
    | **Digital Twin** | Unified view of asset state, historical context for decisions, simulation capability | Direct sensor queries (slow, no historical context), static asset registry (no real-time state) |
    | **TimescaleDB** | PostgreSQL extension for time-series, excellent compression, mature SQL support | InfluxDB (limited SQL), Cassandra (no aggregations), custom solution (high maintenance) |
    | **Kafka Message Broker** | Handle burst traffic, replay capability, decouple producers/consumers | Direct database writes (data loss on failure), RabbitMQ (lower throughput) |
    | **Feature Store** | Centralized ML features, consistent training/inference, feature reuse | Ad-hoc feature computation (inconsistent), duplicate feature logic (maintenance burden) |
    | **Survival Analysis** | Model time-to-failure distribution, handle censored data, probabilistic RUL | Classification only (no time component), regression (no uncertainty quantification) |

    **Key Trade-off:** We chose **edge-first architecture** to reduce cloud bandwidth by 90% and enable sub-100ms anomaly detection, at the cost of deploying and managing 5,000 edge gateways. For manufacturing, real-time detection is critical to prevent cascading failures.

    ---

    ## API Design

    ### 1. Ingest Sensor Data (Edge to Cloud)

    **Request:**
    ```http
    POST /api/v1/telemetry/batch
    Content-Type: application/json
    Authorization: Bearer <gateway_token>

    {
      "gateway_id": "gateway_factory01_line03",
      "timestamp": 1735819200000,
      "readings": [
        {
          "sensor_id": "sensor_motor_A1_vibration",
          "timestamp": 1735819200000,
          "value": 0.45,
          "quality": "good"
        },
        {
          "sensor_id": "sensor_motor_A1_temperature",
          "timestamp": 1735819200000,
          "value": 75.2,
          "quality": "good"
        }
      ],
      "anomalies": [
        {
          "sensor_id": "sensor_motor_A1_vibration",
          "timestamp": 1735819195000,
          "value": 2.8,
          "anomaly_score": 0.95,
          "reason": "Vibration spike detected"
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "status": "accepted",
      "readings_accepted": 2,
      "anomalies_accepted": 1,
      "next_sync": 1735819260000
    }
    ```

    ---

    ### 2. Query Digital Twin State

    **Request:**
    ```http
    POST /api/v1/twin/query
    Content-Type: application/json
    Authorization: Bearer <user_token>

    {
      "query": "query GetMotorState($motorId: ID!) {
        asset(id: $motorId) {
          id
          name
          type
          status
          healthScore
          currentState {
            temperature
            vibration
            rpm
            power
            lastUpdate
          }
          predictedFailure {
            probability
            timeToFailure
            failureMode
            confidence
          }
          maintenanceSchedule {
            nextScheduled
            lastCompleted
            workOrderId
          }
        }
      }",
      "variables": {
        "motorId": "motor_A1_factory01"
      }
    }
    ```

    **Response:**
    ```json
    {
      "data": {
        "asset": {
          "id": "motor_A1_factory01",
          "name": "Production Line A - Motor 1",
          "type": "electric_motor",
          "status": "warning",
          "healthScore": 72,
          "currentState": {
            "temperature": 75.2,
            "vibration": 0.45,
            "rpm": 1800,
            "power": 15.3,
            "lastUpdate": "2025-01-02T14:30:00Z"
          },
          "predictedFailure": {
            "probability": 0.35,
            "timeToFailure": 168,
            "failureMode": "bearing_failure",
            "confidence": 0.82
          },
          "maintenanceSchedule": {
            "nextScheduled": "2025-01-10T08:00:00Z",
            "lastCompleted": "2024-12-15T10:30:00Z",
            "workOrderId": "WO-2024-12-0153"
          }
        }
      }
    }
    ```

    ---

    ### 3. Get Remaining Useful Life (RUL) Prediction

    **Request:**
    ```http
    GET /api/v1/assets/motor_A1_factory01/rul
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "asset_id": "motor_A1_factory01",
      "component": "bearing",
      "rul_prediction": {
        "median_hours": 168,
        "confidence_interval": {
          "lower": 120,
          "upper": 240,
          "confidence": 0.90
        },
        "survival_curve": [
          { "hours": 0, "survival_probability": 1.0 },
          { "hours": 24, "survival_probability": 0.98 },
          { "hours": 72, "survival_probability": 0.90 },
          { "hours": 168, "survival_probability": 0.50 },
          { "hours": 336, "survival_probability": 0.10 }
        ],
        "failure_mode": "bearing_failure",
        "model_version": "v2.3.1",
        "prediction_timestamp": "2025-01-02T14:30:00Z"
      },
      "contributing_factors": [
        {
          "factor": "high_vibration",
          "importance": 0.45,
          "current_value": 0.45,
          "threshold": 0.30
        },
        {
          "factor": "elevated_temperature",
          "importance": 0.30,
          "current_value": 75.2,
          "threshold": 70.0
        },
        {
          "factor": "operating_hours",
          "importance": 0.25,
          "current_value": 8450,
          "threshold": 8000
        }
      ],
      "recommended_actions": [
        {
          "priority": "high",
          "action": "Schedule bearing replacement within 7 days",
          "estimated_downtime_hours": 4,
          "estimated_cost": 2500
        },
        {
          "priority": "medium",
          "action": "Increase inspection frequency to daily",
          "estimated_downtime_hours": 0,
          "estimated_cost": 0
        }
      ]
    }
    ```

    ---

    ### 4. Create Maintenance Work Order

    **Request:**
    ```http
    POST /api/v1/maintenance/workorders
    Content-Type: application/json
    Authorization: Bearer <user_token>

    {
      "asset_id": "motor_A1_factory01",
      "type": "predictive",
      "priority": "high",
      "title": "Bearing replacement - predicted failure",
      "description": "Replace motor bearing due to elevated vibration and predicted failure within 7 days",
      "scheduled_date": "2025-01-09T08:00:00Z",
      "estimated_duration_hours": 4,
      "required_skills": ["mechanical", "electrical"],
      "parts_required": [
        {
          "part_number": "BRG-6308-2RS",
          "quantity": 2,
          "description": "Deep groove ball bearing"
        }
      ],
      "failure_prediction": {
        "probability": 0.35,
        "rul_hours": 168,
        "failure_mode": "bearing_failure"
      }
    }
    ```

    **Response:**
    ```json
    {
      "workorder_id": "WO-2025-01-0089",
      "status": "scheduled",
      "asset_id": "motor_A1_factory01",
      "scheduled_date": "2025-01-09T08:00:00Z",
      "assigned_technician": {
        "id": "tech_001",
        "name": "John Smith",
        "skills": ["mechanical", "electrical"]
      },
      "parts_reserved": true,
      "estimated_cost": {
        "labor": 400,
        "parts": 2500,
        "total": 2900
      },
      "created_at": "2025-01-02T14:35:00Z"
    }
    ```

    ---

    ### 5. Query Sensor Time-Series Data

    **Request:**
    ```http
    GET /api/v1/sensors/sensor_motor_A1_vibration/timeseries?start=2025-01-01T00:00:00Z&end=2025-01-02T00:00:00Z&aggregation=1m
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "sensor_id": "sensor_motor_A1_vibration",
      "unit": "mm/s",
      "aggregation": "1m",
      "data_points": 1440,
      "data": [
        {
          "timestamp": "2025-01-01T00:00:00Z",
          "avg": 0.38,
          "min": 0.35,
          "max": 0.42,
          "stddev": 0.03,
          "count": 60
        },
        {
          "timestamp": "2025-01-01T00:01:00Z",
          "avg": 0.39,
          "min": 0.36,
          "max": 0.43,
          "stddev": 0.03,
          "count": 60
        }
      ],
      "anomalies": [
        {
          "timestamp": "2025-01-01T14:23:00Z",
          "value": 2.8,
          "anomaly_score": 0.95,
          "severity": "high"
        }
      ],
      "statistics": {
        "overall_avg": 0.42,
        "overall_min": 0.30,
        "overall_max": 2.8,
        "overall_stddev": 0.15
      }
    }
    ```

    ---

    ## Database Schema

    ### Time-Series Data (TimescaleDB)

    **Sensor Readings (Hypertable):**

    ```sql
    -- Create hypertable for automatic time-based partitioning
    CREATE TABLE sensor_readings (
        sensor_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        value REAL NOT NULL,
        quality VARCHAR(20) DEFAULT 'good',  -- good, bad, uncertain
        PRIMARY KEY (sensor_id, time)
    );

    -- Convert to hypertable (automatic partitioning by time)
    SELECT create_hypertable('sensor_readings', 'time');

    -- Enable compression (10:1 ratio for industrial data)
    ALTER TABLE sensor_readings SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'sensor_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    -- Compression policy: compress data older than 7 days
    SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');

    -- Retention policy: drop data older than 5 years
    SELECT add_retention_policy('sensor_readings', INTERVAL '5 years');

    -- Indexes
    CREATE INDEX idx_sensor_time ON sensor_readings (sensor_id, time DESC);
    CREATE INDEX idx_quality ON sensor_readings (quality, time DESC) WHERE quality != 'good';
    ```

    **Continuous Aggregations:**

    ```sql
    -- 1-minute aggregation (for medium-term queries)
    CREATE MATERIALIZED VIEW sensor_readings_1min
    WITH (timescaledb.continuous) AS
    SELECT
        sensor_id,
        time_bucket('1 minute', time) AS bucket,
        AVG(value) as avg_value,
        MIN(value) as min_value,
        MAX(value) as max_value,
        STDDEV(value) as stddev_value,
        COUNT(*) as reading_count,
        SUM(CASE WHEN quality = 'good' THEN 1 ELSE 0 END) as good_count
    FROM sensor_readings
    GROUP BY sensor_id, bucket;

    SELECT add_continuous_aggregate_policy('sensor_readings_1min',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 minute',
        schedule_interval => INTERVAL '1 minute'
    );

    -- 1-hour aggregation (for historical analysis)
    CREATE MATERIALIZED VIEW sensor_readings_1hour
    WITH (timescaledb.continuous) AS
    SELECT
        sensor_id,
        time_bucket('1 hour', time) AS bucket,
        AVG(value) as avg_value,
        MIN(value) as min_value,
        MAX(value) as max_value,
        STDDEV(value) as stddev_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value,
        COUNT(*) as reading_count
    FROM sensor_readings
    GROUP BY sensor_id, bucket;

    SELECT add_continuous_aggregate_policy('sensor_readings_1hour',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour'
    );
    ```

    ---

    ### Asset & Twin Data (PostgreSQL)

    **Assets:**

    ```sql
    CREATE TABLE assets (
        asset_id UUID PRIMARY KEY,
        asset_code VARCHAR(100) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        asset_type VARCHAR(100) NOT NULL,     -- motor, pump, compressor, conveyor
        manufacturer VARCHAR(100),
        model VARCHAR(100),
        serial_number VARCHAR(100),
        installation_date DATE,
        factory_id UUID REFERENCES factories(factory_id),
        parent_asset_id UUID REFERENCES assets(asset_id),  -- For hierarchical assets
        location JSONB,                        -- { "building": "A", "floor": 2, "line": 3 }
        specifications JSONB,                  -- Technical specs
        status VARCHAR(20) DEFAULT 'active',   -- active, maintenance, inactive, retired
        health_score INT,                      -- 0-100
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_type (asset_type),
        INDEX idx_factory (factory_id),
        INDEX idx_parent (parent_asset_id),
        INDEX idx_status (status)
    );

    -- Example asset:
    INSERT INTO assets (asset_id, asset_code, name, asset_type, specifications) VALUES (
        'motor_A1_factory01',
        'MTR-A1-001',
        'Production Line A - Motor 1',
        'electric_motor',
        '{
            "rated_power_kw": 15,
            "rated_rpm": 1800,
            "voltage": 480,
            "phases": 3,
            "bearing_type": "6308-2RS",
            "lubrication": "grease",
            "mtbf_hours": 10000
        }'
    );
    ```

    **Sensors:**

    ```sql
    CREATE TABLE sensors (
        sensor_id UUID PRIMARY KEY,
        sensor_code VARCHAR(100) UNIQUE NOT NULL,
        asset_id UUID REFERENCES assets(asset_id),
        sensor_type VARCHAR(50) NOT NULL,      -- temperature, vibration, pressure, flow
        unit VARCHAR(20) NOT NULL,             -- celsius, mm/s, bar, m3/h
        sampling_rate_hz REAL NOT NULL,        -- 1.0, 0.1, 0.01
        normal_range JSONB,                    -- { "min": 0, "max": 100 }
        critical_thresholds JSONB,             -- { "low": 10, "high": 90 }
        protocol VARCHAR(50),                  -- opc_ua, modbus, mqtt
        address VARCHAR(255),                  -- Protocol-specific address
        status VARCHAR(20) DEFAULT 'active',
        last_reading_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_sensors (asset_id),
        INDEX idx_sensor_type (sensor_type),
        INDEX idx_status (status)
    );
    ```

    **Digital Twin State:**

    ```sql
    CREATE TABLE twin_state (
        asset_id UUID PRIMARY KEY REFERENCES assets(asset_id),
        state JSONB NOT NULL,                  -- Current state (all sensor values)
        derived_metrics JSONB,                 -- Calculated metrics (efficiency, etc.)
        last_updated TIMESTAMP NOT NULL,
        version BIGINT NOT NULL,               -- Optimistic locking
        INDEX idx_last_updated (last_updated DESC)
    );

    -- Example twin state:
    UPDATE twin_state SET state = '{
        "temperature": 75.2,
        "vibration": 0.45,
        "rpm": 1800,
        "power": 15.3,
        "current": 22.5,
        "efficiency": 0.92
    }', last_updated = NOW(), version = version + 1
    WHERE asset_id = 'motor_A1_factory01';
    ```

    **Anomalies:**

    ```sql
    CREATE TABLE anomalies (
        anomaly_id UUID PRIMARY KEY,
        sensor_id UUID REFERENCES sensors(sensor_id),
        asset_id UUID REFERENCES assets(asset_id),
        detected_at TIMESTAMP NOT NULL,
        anomaly_type VARCHAR(50) NOT NULL,     -- spike, drift, oscillation, flatline
        severity VARCHAR(20) NOT NULL,         -- low, medium, high, critical
        anomaly_score REAL NOT NULL,           -- 0.0-1.0
        value REAL,
        expected_value REAL,
        deviation REAL,
        reason TEXT,
        detected_by VARCHAR(50),               -- edge_ml, cloud_ml, rule
        status VARCHAR(20) DEFAULT 'open',     -- open, investigating, resolved, false_positive
        resolved_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_anomalies (asset_id, detected_at DESC),
        INDEX idx_sensor_anomalies (sensor_id, detected_at DESC),
        INDEX idx_status (status, detected_at DESC),
        INDEX idx_severity (severity, detected_at DESC)
    );
    ```

    **Failure Predictions:**

    ```sql
    CREATE TABLE failure_predictions (
        prediction_id UUID PRIMARY KEY,
        asset_id UUID REFERENCES assets(asset_id),
        component VARCHAR(100) NOT NULL,       -- bearing, seal, winding, etc.
        failure_mode VARCHAR(100) NOT NULL,    -- bearing_failure, overheating, imbalance
        predicted_at TIMESTAMP NOT NULL,
        failure_probability REAL NOT NULL,     -- 0.0-1.0
        rul_hours_median REAL,                 -- Remaining useful life (median)
        rul_hours_lower REAL,                  -- 90% confidence interval lower bound
        rul_hours_upper REAL,                  -- 90% confidence interval upper bound
        confidence REAL NOT NULL,              -- Model confidence 0.0-1.0
        contributing_factors JSONB,            -- Feature importance
        model_version VARCHAR(50),
        status VARCHAR(20) DEFAULT 'active',   -- active, maintenance_scheduled, resolved
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_predictions (asset_id, predicted_at DESC),
        INDEX idx_active_predictions (status, failure_probability DESC) WHERE status = 'active'
    );
    ```

    **Maintenance Work Orders:**

    ```sql
    CREATE TABLE work_orders (
        workorder_id VARCHAR(50) PRIMARY KEY,
        asset_id UUID REFERENCES assets(asset_id),
        type VARCHAR(50) NOT NULL,             -- predictive, preventive, corrective, inspection
        priority VARCHAR(20) NOT NULL,         -- low, medium, high, critical
        title VARCHAR(255) NOT NULL,
        description TEXT,
        scheduled_date TIMESTAMP,
        completed_date TIMESTAMP,
        estimated_duration_hours REAL,
        actual_duration_hours REAL,
        assigned_technician_id UUID REFERENCES technicians(technician_id),
        status VARCHAR(20) DEFAULT 'scheduled', -- scheduled, in_progress, completed, cancelled
        estimated_cost DECIMAL(10, 2),
        actual_cost DECIMAL(10, 2),
        parts_required JSONB,
        failure_prediction_id UUID REFERENCES failure_predictions(prediction_id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_workorders (asset_id, scheduled_date DESC),
        INDEX idx_status (status, scheduled_date),
        INDEX idx_technician (assigned_technician_id, scheduled_date)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Sensor Data Collection & Edge Processing

    ```mermaid
    sequenceDiagram
        participant Sensor as Sensor
        participant PLC as PLC/DCS
        participant OPC as OPC UA Server
        participant Edge as Edge Gateway
        participant EdgeML as Edge ML Engine
        participant Cloud as Cloud Platform

        Note over Sensor: Continuous sampling<br/>1 Hz (vibration)

        Sensor->>PLC: Analog signal
        PLC->>OPC: Modbus TCP / OPC UA
        OPC->>Edge: OPC UA Subscribe<br/>(data change notification)

        Edge->>Edge: Buffer readings<br/>100ms window

        Edge->>EdgeML: Batch inference<br/>(100 readings)
        EdgeML->>EdgeML: Anomaly detection<br/>Isolation Forest<br/>< 50ms

        alt Anomaly detected
            EdgeML->>Edge: Anomaly alert<br/>Score: 0.95
            Edge->>Cloud: Send alert + context<br/>(10 sec history)
            Cloud-->>Edge: Ack

            Edge->>Edge: Store in local buffer<br/>(for investigation)
        else Normal reading
            EdgeML->>Edge: Normal (score < 0.7)
            Edge->>Edge: Aggregate to 1-min avg

            Note over Edge: Send aggregated data<br/>every 1 minute

            Edge->>Cloud: Send 1-min aggregates<br/>(90% data reduction)
        end

        alt Network failure
            Edge->>Edge: Buffer locally<br/>(24h capacity)
            Note over Edge: Continue edge processing<br/>Alerts via local SCADA
        end
    ```

    ---

    ### Digital Twin State Synchronization

    ```mermaid
    sequenceDiagram
        participant Edge as Edge Gateway
        participant TwinSync as Twin Synchronizer
        participant TwinState as Twin State Store
        participant TwinQuery as Twin Query API
        participant Dashboard as Dashboard

        Note over Edge: State change detected<br/>(RPM: 1800 → 1850)

        Edge->>TwinSync: State update<br/>{ asset_id, rpm: 1850 }
        TwinSync->>TwinSync: Validate update<br/>Check version

        TwinSync->>TwinState: Update state<br/>(optimistic lock)

        alt Update successful
            TwinState-->>TwinSync: Success, version: 12345
            TwinSync-->>Edge: Ack

            TwinState->>TwinState: Trigger webhooks<br/>(state change subscribers)
        else Version conflict
            TwinState-->>TwinSync: Conflict, current version: 12346
            TwinSync->>TwinState: Retry with latest version
        end

        Dashboard->>TwinQuery: Subscribe to asset updates<br/>WebSocket
        TwinState->>TwinQuery: Notify state change
        TwinQuery->>Dashboard: Push update<br/>{ rpm: 1850 }

        Dashboard->>Dashboard: Update UI<br/>Real-time gauge
    ```

    ---

    ### Predictive Maintenance Flow

    ```mermaid
    sequenceDiagram
        participant Sensor as Sensors
        participant TSDB as Time-Series DB
        participant FeatureStore as Feature Store
        participant MLInference as ML Inference
        participant RULEngine as RUL Engine
        participant Scheduler as Maintenance Scheduler
        participant CMMS as CMMS

        Note over Sensor: Continuous monitoring<br/>Vibration, Temp, etc.

        Sensor->>TSDB: Store readings

        Note over FeatureStore: Hourly feature extraction

        TSDB->>FeatureStore: Query raw data<br/>(last 7 days)
        FeatureStore->>FeatureStore: Extract features:<br/>- RMS vibration<br/>- Temp trend<br/>- Operating hours<br/>- Load cycles

        FeatureStore->>MLInference: Request prediction<br/>(feature vector)
        MLInference->>MLInference: Load model<br/>(bearing failure model)
        MLInference->>MLInference: Predict failure probability<br/>Output: 0.35

        alt Failure probability > 0.3
            MLInference->>RULEngine: Compute RUL<br/>(survival analysis)
            RULEngine->>RULEngine: Cox proportional hazard<br/>Median RUL: 168 hours<br/>CI: [120, 240]

            RULEngine->>Scheduler: Create prediction record<br/>RUL: 168h, Prob: 0.35

            Scheduler->>Scheduler: Check maintenance policy<br/>Threshold: RUL < 240h
            Scheduler->>Scheduler: Calculate optimal time<br/>(minimize cost + downtime)

            Scheduler->>CMMS: Create work order<br/>Type: Predictive<br/>Schedule: +7 days
            CMMS-->>Scheduler: WO-2025-01-0089

            CMMS->>CMMS: Reserve parts<br/>Assign technician<br/>Block production slot
        else Failure probability < 0.3
            MLInference->>MLInference: Continue monitoring
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 OPC UA Data Collection

    ```python
    from opcua import Client, ua
    import asyncio
    from typing import List, Dict, Any
    import logging

    logger = logging.getLogger(__name__)

    class OPCUACollector:
        """
        OPC UA data collector for industrial sensors

        OPC UA (Open Platform Communications Unified Architecture):
        - Standard industrial communication protocol
        - Secure (encryption, authentication, authorization)
        - Platform-independent (vendor-neutral)
        - Supports complex data types and discovery

        Collection strategies:
        1. Subscription: Server notifies on data change (efficient)
        2. Polling: Client requests data periodically (simple)
        3. Historical access: Query past data (analysis)
        """

        def __init__(self, server_url: str, namespace: str):
            self.server_url = server_url
            self.namespace = namespace
            self.client = None
            self.subscription = None
            self.monitored_items = {}

        async def connect(self):
            """Connect to OPC UA server with security"""
            self.client = Client(self.server_url)

            # Configure security (certificate-based)
            self.client.set_security_string(
                "Basic256Sha256,SignAndEncrypt," +
                "certificate.pem,private_key.pem"
            )

            await self.client.connect()
            logger.info(f"Connected to OPC UA server: {self.server_url}")

            # Browse server namespace
            root = self.client.get_root_node()
            logger.info(f"Server namespace index: {self.namespace}")

        async def subscribe_sensors(self, sensor_configs: List[Dict]):
            """
            Subscribe to sensor data changes

            OPC UA subscription advantages:
            - Server pushes data only when it changes (efficient)
            - Configurable deadband (filter small changes)
            - Queue overflow handling
            - Monitored item sampling interval
            """
            # Create subscription (100ms publishing interval)
            self.subscription = await self.client.create_subscription(
                period=100,  # 100ms = 10 Hz max
                handler=DataChangeHandler(self)
            )

            for config in sensor_configs:
                # Construct node ID from sensor config
                node_id = f"ns={self.namespace};s={config['opc_address']}"
                node = self.client.get_node(node_id)

                # Create monitored item with deadband filter
                # Deadband: Only report if value changes by > threshold
                deadband = config.get('deadband', 0.01)  # 1% default

                params = ua.MonitoredItemCreateRequest()
                params.ItemToMonitor.NodeId = node.nodeid
                params.MonitoringMode = ua.MonitoringMode.Reporting
                params.RequestedParameters.ClientHandle = config['sensor_id']
                params.RequestedParameters.SamplingInterval = 1000 / config['sampling_rate_hz']
                params.RequestedParameters.QueueSize = 10
                params.RequestedParameters.DiscardOldest = True

                # Add deadband filter (percentage)
                params.RequestedParameters.Filter = ua.DataChangeFilter()
                params.RequestedParameters.Filter.Trigger = ua.DataChangeTrigger.StatusValue
                params.RequestedParameters.Filter.DeadbandType = ua.DeadbandType.Percent
                params.RequestedParameters.Filter.DeadbandValue = deadband * 100

                handle = await self.subscription.subscribe_data_change(node, params)

                self.monitored_items[config['sensor_id']] = {
                    'handle': handle,
                    'node': node,
                    'config': config
                }

                logger.info(f"Subscribed to sensor: {config['sensor_id']}")

        async def read_historical_data(
            self,
            sensor_id: str,
            start_time: datetime,
            end_time: datetime
        ) -> List[Dict]:
            """
            Read historical data from OPC UA server

            Use case: Backfill missing data, ML training
            """
            config = self.monitored_items[sensor_id]['config']
            node = self.monitored_items[sensor_id]['node']

            # OPC UA historical read request
            start = ua.get_win_epoch(start_time)
            end = ua.get_win_epoch(end_time)

            history = await node.read_raw_history(start, end)

            readings = []
            for datavalue in history:
                readings.append({
                    'sensor_id': sensor_id,
                    'timestamp': datavalue.SourceTimestamp.timestamp() * 1000,
                    'value': datavalue.Value.Value,
                    'quality': self._map_quality(datavalue.StatusCode)
                })

            logger.info(f"Retrieved {len(readings)} historical readings for {sensor_id}")
            return readings

        def _map_quality(self, status_code: ua.StatusCode) -> str:
            """Map OPC UA status code to quality"""
            if status_code.is_good():
                return 'good'
            elif status_code.is_uncertain():
                return 'uncertain'
            else:
                return 'bad'


    class DataChangeHandler:
        """Handler for OPC UA data change notifications"""

        def __init__(self, collector: OPCUACollector):
            self.collector = collector
            self.edge_processor = EdgeProcessor()

        def datachange_notification(self, node, val, data):
            """
            Called when monitored item value changes

            This runs in subscription thread - must be fast!
            """
            sensor_id = data.ClientHandle

            reading = {
                'sensor_id': sensor_id,
                'timestamp': data.MonitoredItem.Value.SourceTimestamp.timestamp() * 1000,
                'value': val,
                'quality': self._map_quality(data.MonitoredItem.Value.StatusCode)
            }

            # Forward to edge processor (async, non-blocking)
            asyncio.create_task(self.edge_processor.process_reading(reading))


    class EdgeProcessor:
        """
        Edge processing pipeline

        Pipeline stages:
        1. Validation: Range checks, quality checks
        2. Buffering: Time-based windows for batch processing
        3. Anomaly detection: ML-based edge inference
        4. Aggregation: Reduce data volume
        5. Cloud sync: Send filtered/aggregated data
        """

        def __init__(self):
            self.buffer = {}  # sensor_id -> list of readings
            self.buffer_size = 100  # 100 readings per batch
            self.anomaly_detector = EdgeAnomalyDetector()
            self.cloud_sync = CloudSync()

        async def process_reading(self, reading: Dict):
            """Process incoming sensor reading"""
            sensor_id = reading['sensor_id']

            # Stage 1: Validation
            if not self._validate_reading(reading):
                logger.warning(f"Invalid reading from {sensor_id}: {reading}")
                return

            # Stage 2: Buffering
            if sensor_id not in self.buffer:
                self.buffer[sensor_id] = []
            self.buffer[sensor_id].append(reading)

            # Stage 3: Batch processing when buffer full
            if len(self.buffer[sensor_id]) >= self.buffer_size:
                await self._process_batch(sensor_id)

        async def _process_batch(self, sensor_id: str):
            """Process batch of readings"""
            batch = self.buffer[sensor_id]

            # Stage 3: Anomaly detection
            anomalies = await self.anomaly_detector.detect_batch(sensor_id, batch)

            if anomalies:
                # High priority: Send anomalies immediately with context
                await self.cloud_sync.send_anomalies(sensor_id, anomalies, batch)

            # Stage 4: Aggregation (reduce 100 readings to 1 aggregate)
            aggregate = self._aggregate_batch(batch)

            # Stage 5: Cloud sync (send aggregate)
            await self.cloud_sync.send_aggregate(sensor_id, aggregate)

            # Clear buffer
            self.buffer[sensor_id] = []

        def _validate_reading(self, reading: Dict) -> bool:
            """Validate sensor reading"""
            # Check quality
            if reading['quality'] != 'good':
                return False

            # Check range (from sensor config)
            # In production, load from database
            normal_range = {'min': -1000, 'max': 1000}
            if not (normal_range['min'] <= reading['value'] <= normal_range['max']):
                return False

            return True

        def _aggregate_batch(self, batch: List[Dict]) -> Dict:
            """Aggregate batch of readings"""
            values = [r['value'] for r in batch]

            return {
                'timestamp': batch[-1]['timestamp'],  # Last reading timestamp
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'stddev': np.std(values),
                'count': len(values)
            }
    ```

    **OPC UA Benefits for Industrial IoT:**

    ```
    Advantages over other protocols:

    1. Security:
       - X.509 certificate authentication
       - AES-256 encryption
       - Fine-grained access control
       - Audit trail

    2. Vendor-neutral:
       - Works with all major PLC vendors (Siemens, Rockwell, Schneider)
       - Standardized by OPC Foundation
       - No vendor lock-in

    3. Complex data types:
       - Structures, arrays, enums
       - Type definitions
       - Object-oriented data model

    4. Discovery:
       - Browse server namespace
       - Find available sensors dynamically
       - No hard-coded addresses

    5. Historical access:
       - Query past data
       - Aggregates (min, max, avg)
       - Useful for ML training

    Performance characteristics:
    - Latency: 10-50ms (local network)
    - Throughput: 100K+ values/sec (single server)
    - Overhead: ~30% vs. raw TCP (acceptable for benefits)
    ```

    ---

    ## 3.2 Edge Anomaly Detection

    ```python
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from typing import List, Dict
    import pickle

    class EdgeAnomalyDetector:
        """
        Edge ML for real-time anomaly detection

        Why edge ML:
        1. Low latency: < 100ms detection (vs. 500ms+ cloud round-trip)
        2. Network resilience: Works during cloud outages
        3. Bandwidth savings: Only send anomalies to cloud (90% reduction)
        4. Privacy: Sensitive data stays on-premises

        Techniques:
        1. Isolation Forest: Fast, unsupervised, handles high dimensions
        2. Statistical methods: 3-sigma, moving average
        3. Domain rules: Threshold violations

        Model deployment:
        - Train in cloud (GPU cluster, large dataset)
        - Deploy to edge (small model, CPU inference)
        - Update weekly via OTA
        """

        def __init__(self):
            self.models = {}  # sensor_id -> model
            self.baselines = {}  # sensor_id -> statistical baseline
            self.model_dir = '/opt/edge-ml/models'
            self._load_models()

        def _load_models(self):
            """Load pre-trained models from disk"""
            # In production, models deployed from cloud via OTA
            for model_file in os.listdir(self.model_dir):
                sensor_id = model_file.replace('.pkl', '')
                with open(f"{self.model_dir}/{model_file}", 'rb') as f:
                    self.models[sensor_id] = pickle.load(f)
                logger.info(f"Loaded model for {sensor_id}")

        async def detect_batch(
            self,
            sensor_id: str,
            readings: List[Dict]
        ) -> List[Dict]:
            """
            Detect anomalies in batch of readings

            Multi-level detection:
            1. Rule-based (instant, high precision)
            2. Statistical (fast, good recall)
            3. ML-based (slower, best accuracy)
            """
            anomalies = []

            # Extract values
            values = np.array([r['value'] for r in readings])
            timestamps = [r['timestamp'] for r in readings]

            # Level 1: Rule-based detection
            rule_anomalies = self._detect_rules(sensor_id, values, timestamps)
            anomalies.extend(rule_anomalies)

            # Level 2: Statistical detection
            stat_anomalies = self._detect_statistical(sensor_id, values, timestamps)
            anomalies.extend(stat_anomalies)

            # Level 3: ML-based detection (if model available)
            if sensor_id in self.models:
                ml_anomalies = self._detect_ml(sensor_id, values, timestamps)
                anomalies.extend(ml_anomalies)

            # Deduplicate (same reading flagged by multiple methods)
            anomalies = self._deduplicate_anomalies(anomalies)

            return anomalies

        def _detect_rules(
            self,
            sensor_id: str,
            values: np.ndarray,
            timestamps: List[int]
        ) -> List[Dict]:
            """
            Rule-based anomaly detection

            Rules (loaded from config):
            1. Absolute thresholds (critical high/low)
            2. Rate of change (spike detection)
            3. Flatline (sensor stuck)
            """
            anomalies = []

            # Load sensor config (thresholds)
            config = self._get_sensor_config(sensor_id)
            critical_high = config.get('critical_high')
            critical_low = config.get('critical_low')

            # Check absolute thresholds
            for i, value in enumerate(values):
                if critical_high and value > critical_high:
                    anomalies.append({
                        'timestamp': timestamps[i],
                        'value': value,
                        'anomaly_type': 'critical_high',
                        'severity': 'critical',
                        'anomaly_score': 1.0,
                        'reason': f'Value {value} exceeds critical threshold {critical_high}',
                        'detection_method': 'rule'
                    })
                elif critical_low and value < critical_low:
                    anomalies.append({
                        'timestamp': timestamps[i],
                        'value': value,
                        'anomaly_type': 'critical_low',
                        'severity': 'critical',
                        'anomaly_score': 1.0,
                        'reason': f'Value {value} below critical threshold {critical_low}',
                        'detection_method': 'rule'
                    })

            # Check rate of change (spike)
            if len(values) > 1:
                rate_of_change = np.diff(values)
                max_rate = config.get('max_rate_of_change', np.inf)

                for i, rate in enumerate(rate_of_change):
                    if abs(rate) > max_rate:
                        anomalies.append({
                            'timestamp': timestamps[i+1],
                            'value': values[i+1],
                            'anomaly_type': 'spike',
                            'severity': 'high',
                            'anomaly_score': min(abs(rate) / max_rate, 1.0),
                            'reason': f'Rate of change {rate:.2f} exceeds max {max_rate}',
                            'detection_method': 'rule'
                        })

            # Check flatline (no variation)
            if len(values) > 10:
                if np.std(values) < config.get('min_stddev', 0.001):
                    anomalies.append({
                        'timestamp': timestamps[-1],
                        'value': values[-1],
                        'anomaly_type': 'flatline',
                        'severity': 'medium',
                        'anomaly_score': 0.8,
                        'reason': 'Sensor shows no variation (possible stuck)',
                        'detection_method': 'rule'
                    })

            return anomalies

        def _detect_statistical(
            self,
            sensor_id: str,
            values: np.ndarray,
            timestamps: List[int]
        ) -> List[Dict]:
            """
            Statistical anomaly detection

            Method: Modified Z-score (robust to outliers)
            - Normal: |z| < 3 (99.7% of data)
            - Anomaly: |z| > 3.5 (tighter than 3-sigma)
            """
            anomalies = []

            # Get or create baseline
            if sensor_id not in self.baselines:
                self.baselines[sensor_id] = {
                    'median': np.median(values),
                    'mad': self._median_absolute_deviation(values)
                }

            baseline = self.baselines[sensor_id]
            median = baseline['median']
            mad = baseline['mad']

            # Modified Z-score
            # z = 0.6745 * (x - median) / MAD
            # More robust than standard z-score
            modified_z_scores = 0.6745 * (values - median) / mad if mad > 0 else np.zeros_like(values)

            # Detect anomalies (|z| > 3.5)
            anomaly_indices = np.where(np.abs(modified_z_scores) > 3.5)[0]

            for i in anomaly_indices:
                z_score = modified_z_scores[i]
                anomalies.append({
                    'timestamp': timestamps[i],
                    'value': values[i],
                    'anomaly_type': 'statistical',
                    'severity': 'high' if abs(z_score) > 5 else 'medium',
                    'anomaly_score': min(abs(z_score) / 10, 1.0),
                    'reason': f'Modified Z-score: {z_score:.2f} (expected ±3.5)',
                    'detection_method': 'statistical',
                    'expected_value': median
                })

            # Update baseline (exponential moving average)
            alpha = 0.1  # Smoothing factor
            self.baselines[sensor_id] = {
                'median': alpha * np.median(values) + (1 - alpha) * median,
                'mad': alpha * self._median_absolute_deviation(values) + (1 - alpha) * mad
            }

            return anomalies

        def _detect_ml(
            self,
            sensor_id: str,
            values: np.ndarray,
            timestamps: List[int]
        ) -> List[Dict]:
            """
            ML-based anomaly detection using Isolation Forest

            Isolation Forest:
            - Isolates anomalies by randomly selecting features and split values
            - Anomalies are easier to isolate (require fewer splits)
            - Fast training and inference (O(log n))
            - No need for labeled data (unsupervised)

            Features:
            1. Current value
            2. Moving averages (5, 20, 60 samples)
            3. Standard deviation (rolling window)
            4. Rate of change
            5. FFT features (for vibration signals)
            """
            model = self.models[sensor_id]

            # Extract features
            features = self._extract_features(values)

            # Predict anomaly scores (-1 = anomaly, 1 = normal)
            predictions = model.predict(features)
            anomaly_scores = model.decision_function(features)

            # decision_function returns negative values for anomalies
            # Normalize to 0-1 range (higher = more anomalous)
            anomaly_scores = 1 / (1 + np.exp(anomaly_scores))  # Sigmoid

            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1 and score > 0.7:  # Anomaly threshold
                    anomalies.append({
                        'timestamp': timestamps[i],
                        'value': values[i],
                        'anomaly_type': 'ml_detected',
                        'severity': 'high' if score > 0.9 else 'medium',
                        'anomaly_score': float(score),
                        'reason': f'ML model detected anomalous pattern (score: {score:.2f})',
                        'detection_method': 'ml'
                    })

            return anomalies

        def _extract_features(self, values: np.ndarray) -> np.ndarray:
            """
            Extract features for ML model

            Feature engineering for time-series:
            - Current value and recent history
            - Statistical moments (mean, std, skew, kurtosis)
            - Trend indicators
            - Frequency domain (FFT for vibration)
            """
            n = len(values)
            features = []

            for i in range(n):
                # Window: last 20 samples (or less if near start)
                window_size = min(20, i + 1)
                window = values[max(0, i - window_size + 1):i + 1]

                feature_vector = [
                    values[i],                          # Current value
                    np.mean(window),                    # Moving average
                    np.std(window) if len(window) > 1 else 0,  # Std dev
                    np.max(window) - np.min(window),    # Range
                    values[i] - np.mean(window) if len(window) > 1 else 0,  # Deviation from MA
                ]

                # Rate of change
                if i > 0:
                    feature_vector.append(values[i] - values[i-1])
                else:
                    feature_vector.append(0)

                features.append(feature_vector)

            return np.array(features)

        def _median_absolute_deviation(self, values: np.ndarray) -> float:
            """
            Calculate Median Absolute Deviation (MAD)

            MAD is more robust to outliers than standard deviation
            MAD = median(|x_i - median(x)|)
            """
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            return mad if mad > 0 else 0.001  # Avoid division by zero

        def _deduplicate_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
            """Deduplicate anomalies detected by multiple methods"""
            if not anomalies:
                return []

            # Group by timestamp
            by_timestamp = {}
            for anom in anomalies:
                ts = anom['timestamp']
                if ts not in by_timestamp:
                    by_timestamp[ts] = []
                by_timestamp[ts].append(anom)

            # Keep highest score for each timestamp
            deduped = []
            for ts, anoms in by_timestamp.items():
                best = max(anoms, key=lambda a: a['anomaly_score'])
                # Combine detection methods
                methods = ','.join(set(a['detection_method'] for a in anoms))
                best['detection_method'] = methods
                deduped.append(best)

            return deduped

        def _get_sensor_config(self, sensor_id: str) -> Dict:
            """Load sensor configuration"""
            # In production, load from database or config file
            return {
                'critical_high': 100.0,
                'critical_low': 0.0,
                'max_rate_of_change': 10.0,
                'min_stddev': 0.001
            }
    ```

    **Edge ML Performance:**

    ```
    Isolation Forest inference (100 samples):
    - Feature extraction: 5ms
    - Model inference: 8ms
    - Total latency: 13ms

    Memory footprint:
    - Model size: 2-5 MB (compressed)
    - Runtime memory: 20-50 MB
    - Fits on edge gateway (16 GB RAM)

    Accuracy (validation):
    - True positive rate: 92% (detects 92% of real anomalies)
    - False positive rate: 3% (3% false alarms)
    - Better than rule-based: 78% TPR, 8% FPR

    Model update frequency:
    - Retrain weekly in cloud (new failure data)
    - Deploy to edge via OTA (overnight)
    - Gradual rollout (A/B test 10% of gateways first)
    ```

    ---

    ## 3.3 Predictive Maintenance with Survival Analysis

    ```python
    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from typing import Dict, List, Tuple
    import pickle

    class PredictiveMaintenanceEngine:
        """
        Predictive maintenance using survival analysis

        Why survival analysis:
        1. Models time-to-failure (not just "will fail")
        2. Handles censored data (equipment still running)
        3. Provides uncertainty estimates (confidence intervals)
        4. Interprets feature importance (hazard ratios)

        Cox Proportional Hazards Model:
        - h(t|X) = h0(t) * exp(β1*X1 + β2*X2 + ... + βn*Xn)
        - h(t|X): Hazard rate (instantaneous failure risk)
        - h0(t): Baseline hazard (time-dependent)
        - X: Covariates (features: vibration, temp, age, etc.)
        - β: Coefficients (learned from training data)

        Remaining Useful Life (RUL):
        - Median time until failure
        - Survival curve: P(survival beyond time t)
        - Confidence intervals: Uncertainty bounds
        """

        def __init__(self):
            self.models = {}  # failure_mode -> model
            self.feature_store = FeatureStore()
            self.model_dir = '/opt/ml/models'
            self._load_models()

        def _load_models(self):
            """Load pre-trained models"""
            for model_file in os.listdir(self.model_dir):
                failure_mode = model_file.replace('_cox_model.pkl', '')
                with open(f"{self.model_dir}/{model_file}", 'rb') as f:
                    self.models[failure_mode] = pickle.load(f)
                logger.info(f"Loaded model for {failure_mode}")

        def train_model(
            self,
            failure_mode: str,
            training_data: pd.DataFrame
        ) -> CoxPHFitter:
            """
            Train Cox proportional hazards model

            Training data format:
            - duration: Time to failure (or censoring time) in hours
            - event: 1 if failed, 0 if censored (still running)
            - features: Vibration, temperature, age, cycles, etc.

            Example:
            | asset_id | duration | event | vibration_rms | temp_max | age_hours | load_cycles |
            |----------|----------|-------|---------------|----------|-----------|-------------|
            | motor_01 | 8450     | 1     | 0.52          | 78.5     | 8450      | 12500       |
            | motor_02 | 9200     | 0     | 0.38          | 72.1     | 9200      | 14200       | (censored)
            """
            logger.info(f"Training model for {failure_mode}")
            logger.info(f"Training samples: {len(training_data)}")
            logger.info(f"Failure events: {training_data['event'].sum()}")
            logger.info(f"Censored: {(~training_data['event'].astype(bool)).sum()}")

            # Initialize Cox model
            cph = CoxPHFitter(penalizer=0.1)  # L2 regularization

            # Fit model
            cph.fit(
                training_data,
                duration_col='duration',
                event_col='event',
                show_progress=True
            )

            # Model summary
            logger.info(f"\nModel summary:\n{cph.summary}")
            logger.info(f"Concordance index: {cph.concordance_index_:.3f}")

            # Feature importance (hazard ratios)
            hazard_ratios = np.exp(cph.params_)
            logger.info(f"\nHazard ratios (impact on failure risk):")
            for feature, hr in hazard_ratios.items():
                logger.info(f"  {feature}: {hr:.3f} " +
                          f"({'+' if hr > 1 else ''}{(hr-1)*100:.1f}% per unit increase)")

            # Save model
            model_path = f"{self.model_dir}/{failure_mode}_cox_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(cph, f)
            logger.info(f"Saved model to {model_path}")

            return cph

        def predict_rul(
            self,
            asset_id: str,
            failure_mode: str,
            confidence: float = 0.90
        ) -> Dict:
            """
            Predict Remaining Useful Life (RUL)

            Returns:
            - Median RUL (50th percentile)
            - Confidence interval (e.g., 90% CI)
            - Survival curve (probability of surviving to time t)
            - Failure probability at specific horizons
            """
            if failure_mode not in self.models:
                raise ValueError(f"No model for failure mode: {failure_mode}")

            model = self.models[failure_mode]

            # Get current features for asset
            features = self.feature_store.get_features(asset_id, failure_mode)

            # Create feature vector (single row DataFrame)
            X = pd.DataFrame([features])

            # Predict survival function
            # S(t|X) = P(survival beyond time t | features X)
            survival_function = model.predict_survival_function(X)

            # Extract survival curve for this asset
            survival_curve = survival_function.iloc[:, 0]  # First (only) column

            # Compute median RUL (50% survival probability)
            median_rul = self._compute_percentile_rul(survival_curve, 0.5)

            # Compute confidence interval
            alpha = 1 - confidence
            rul_lower = self._compute_percentile_rul(survival_curve, alpha / 2)
            rul_upper = self._compute_percentile_rul(survival_curve, 1 - alpha / 2)

            # Failure probability at specific horizons
            horizons = [24, 72, 168, 336, 720]  # 1 day, 3 days, 1 week, 2 weeks, 1 month
            failure_probabilities = {}
            for h in horizons:
                # P(failure within h hours) = 1 - P(survival beyond h hours)
                if h in survival_curve.index:
                    survival_prob = survival_curve.loc[h]
                else:
                    # Interpolate if exact time not in index
                    survival_prob = self._interpolate_survival(survival_curve, h)
                failure_probabilities[h] = 1 - survival_prob

            # Feature importance for this prediction
            contributing_factors = self._compute_feature_importance(model, features)

            return {
                'asset_id': asset_id,
                'failure_mode': failure_mode,
                'rul_hours_median': median_rul,
                'rul_hours_lower': rul_lower,
                'rul_hours_upper': rul_upper,
                'confidence': confidence,
                'survival_curve': self._format_survival_curve(survival_curve),
                'failure_probabilities': failure_probabilities,
                'contributing_factors': contributing_factors,
                'model_version': model._model_version if hasattr(model, '_model_version') else 'unknown',
                'prediction_timestamp': pd.Timestamp.now().isoformat()
            }

        def _compute_percentile_rul(
            self,
            survival_curve: pd.Series,
            percentile: float
        ) -> float:
            """
            Compute RUL at given percentile

            Example: percentile=0.5 returns median RUL
            """
            # Find time when survival probability drops below (1 - percentile)
            target_survival = 1 - percentile

            # Survival curve is monotonically decreasing
            # Find first time where S(t) <= target_survival
            below_target = survival_curve <= target_survival

            if below_target.any():
                # Found crossing point
                rul = survival_curve[below_target].index[0]
            else:
                # Never crosses (very reliable asset)
                rul = survival_curve.index[-1]  # Max time in curve

            return float(rul)

        def _interpolate_survival(
            self,
            survival_curve: pd.Series,
            time: float
        ) -> float:
            """Interpolate survival probability at specific time"""
            # Find surrounding times
            times = survival_curve.index

            if time <= times[0]:
                return survival_curve.iloc[0]
            if time >= times[-1]:
                return survival_curve.iloc[-1]

            # Linear interpolation
            idx_after = np.searchsorted(times, time)
            idx_before = idx_after - 1

            t_before = times[idx_before]
            t_after = times[idx_after]
            s_before = survival_curve.iloc[idx_before]
            s_after = survival_curve.iloc[idx_after]

            # Interpolate
            weight = (time - t_before) / (t_after - t_before)
            survival_prob = s_before + weight * (s_after - s_before)

            return float(survival_prob)

        def _format_survival_curve(
            self,
            survival_curve: pd.Series,
            num_points: int = 20
        ) -> List[Dict]:
            """Format survival curve for API response"""
            # Sample curve at regular intervals
            times = np.linspace(
                survival_curve.index[0],
                survival_curve.index[-1],
                num_points
            )

            curve_points = []
            for t in times:
                s = self._interpolate_survival(survival_curve, t)
                curve_points.append({
                    'hours': float(t),
                    'survival_probability': float(s)
                })

            return curve_points

        def _compute_feature_importance(
            self,
            model: CoxPHFitter,
            features: Dict
        ) -> List[Dict]:
            """
            Compute feature importance for this prediction

            Importance = |coefficient * feature_value|
            Higher magnitude = more impact on this prediction
            """
            coefficients = model.params_

            importance_list = []
            for feature, value in features.items():
                if feature in coefficients:
                    coef = coefficients[feature]
                    importance = abs(coef * value)

                    importance_list.append({
                        'feature': feature,
                        'value': value,
                        'coefficient': float(coef),
                        'importance': float(importance)
                    })

            # Sort by importance (descending)
            importance_list.sort(key=lambda x: x['importance'], reverse=True)

            # Normalize to sum to 1
            total_importance = sum(x['importance'] for x in importance_list)
            if total_importance > 0:
                for item in importance_list:
                    item['importance_normalized'] = item['importance'] / total_importance

            return importance_list


    class FeatureStore:
        """
        Feature store for ML features

        Features for predictive maintenance:
        1. Sensor features:
           - RMS vibration (last 24h, 7d, 30d)
           - Max temperature (last 24h, 7d, 30d)
           - Pressure variance
           - Flow rate average

        2. Derived features:
           - Operating hours (total)
           - Load cycles (count)
           - Start/stop cycles
           - Maintenance history (time since last maintenance)

        3. Context features:
           - Ambient temperature
           - Production volume
           - Shift patterns
        """

        def __init__(self, tsdb_connection):
            self.tsdb = tsdb_connection

        def get_features(self, asset_id: str, failure_mode: str) -> Dict:
            """
            Extract features for asset

            Time windows:
            - 24h: Recent trends
            - 7d: Weekly patterns
            - 30d: Monthly trends
            """
            features = {}

            # Sensor features (vibration, temperature, etc.)
            sensor_features = self._get_sensor_features(asset_id)
            features.update(sensor_features)

            # Derived features (age, cycles, etc.)
            derived_features = self._get_derived_features(asset_id)
            features.update(derived_features)

            # Context features
            context_features = self._get_context_features(asset_id)
            features.update(context_features)

            return features

        def _get_sensor_features(self, asset_id: str) -> Dict:
            """Query sensor features from time-series DB"""
            query = """
                SELECT
                    -- 24 hour features
                    AVG(CASE WHEN sensor_type = 'vibration' AND time > NOW() - INTERVAL '24 hours'
                        THEN value END) as vibration_24h_avg,
                    STDDEV(CASE WHEN sensor_type = 'vibration' AND time > NOW() - INTERVAL '24 hours'
                        THEN value END) as vibration_24h_std,
                    MAX(CASE WHEN sensor_type = 'temperature' AND time > NOW() - INTERVAL '24 hours'
                        THEN value END) as temp_24h_max,

                    -- 7 day features
                    AVG(CASE WHEN sensor_type = 'vibration' AND time > NOW() - INTERVAL '7 days'
                        THEN value END) as vibration_7d_avg,
                    MAX(CASE WHEN sensor_type = 'temperature' AND time > NOW() - INTERVAL '7 days'
                        THEN value END) as temp_7d_max,

                    -- 30 day features
                    AVG(CASE WHEN sensor_type = 'vibration' AND time > NOW() - INTERVAL '30 days'
                        THEN value END) as vibration_30d_avg,
                    MAX(CASE WHEN sensor_type = 'temperature' AND time > NOW() - INTERVAL '30 days'
                        THEN value END) as temp_30d_max
                FROM sensor_readings sr
                JOIN sensors s ON sr.sensor_id = s.sensor_id
                WHERE s.asset_id = %s
            """

            result = self.tsdb.execute(query, [asset_id])[0]

            return {k: float(v) if v is not None else 0.0
                   for k, v in result.items()}

        def _get_derived_features(self, asset_id: str) -> Dict:
            """Compute derived features"""
            # In production, these would be maintained in a feature table
            return {
                'age_hours': 8450.0,           # Total operating hours
                'load_cycles': 12500.0,        # Number of start/stop cycles
                'hours_since_maintenance': 720.0,  # Time since last PM
                'maintenance_count': 8.0       # Historical maintenance events
            }

        def _get_context_features(self, asset_id: str) -> Dict:
            """Get contextual features"""
            return {
                'ambient_temperature': 25.0,   # Factory ambient temp
                'production_volume_7d': 15000.0,  # Recent production volume
                'shift_pattern': 2.0           # 1=single, 2=double, 3=triple shift
            }
    ```

    **Predictive Maintenance Results:**

    ```
    Example RUL Prediction for Motor Bearing:

    Current state:
    - Vibration RMS: 0.52 mm/s (normal: < 0.30)
    - Max temperature: 78.5°C (normal: < 70°C)
    - Operating hours: 8,450h (MTBF: 10,000h)
    - Load cycles: 12,500 (high)

    Prediction:
    - Failure mode: Bearing failure
    - Median RUL: 168 hours (7 days)
    - 90% CI: [120, 240] hours (5-10 days)
    - Failure probability:
      - 1 day: 5%
      - 3 days: 18%
      - 7 days: 50%
      - 14 days: 85%

    Contributing factors:
    1. High vibration (45% importance)
    2. Elevated temperature (30% importance)
    3. Operating hours (25% importance)

    Recommended action:
    - Schedule bearing replacement within 7 days
    - Estimated downtime: 4 hours
    - Estimated cost: $2,500 (parts + labor)

    Cost avoidance:
    - Unplanned failure cost: $250K (24h downtime + emergency repair)
    - Planned maintenance cost: $2.5K
    - Savings: $247.5K (99% reduction)
    ```

    ---

    ## 3.4 Digital Twin State Synchronization

    ```python
    import asyncio
    from typing import Dict, Any, Optional
    import json
    import hashlib

    class DigitalTwinEngine:
        """
        Digital Twin: Virtual representation of physical assets

        Digital Twin Components:
        1. State: Current sensor values and derived metrics
        2. Model: Physics-based or ML models of behavior
        3. History: Time-series of past states
        4. Predictions: Future states (RUL, performance)

        State synchronization:
        - Edge → Cloud: Real-time sensor updates
        - Cloud → Edge: Configuration, control commands
        - Optimistic locking: Handle concurrent updates
        - Event-driven: Notify subscribers of changes
        """

        def __init__(self, state_store, event_bus):
            self.state_store = state_store  # Database for twin state
            self.event_bus = event_bus      # Pub/sub for notifications
            self.cache = {}                 # In-memory cache for hot twins

        async def update_twin_state(
            self,
            asset_id: str,
            updates: Dict[str, Any],
            source: str = 'edge'
        ) -> Dict:
            """
            Update digital twin state

            Flow:
            1. Validate update
            2. Load current state
            3. Apply updates (with optimistic locking)
            4. Compute derived metrics
            5. Save new state
            6. Publish event
            """
            # Load current state
            current_state = await self.get_twin_state(asset_id)

            if not current_state:
                raise ValueError(f"Twin not found: {asset_id}")

            # Optimistic locking: Check version
            current_version = current_state.get('version', 0)
            expected_version = updates.get('expected_version')

            if expected_version is not None and expected_version != current_version:
                raise ConflictError(
                    f"Version mismatch: expected {expected_version}, " +
                    f"current {current_version}"
                )

            # Merge updates into current state
            new_state = self._merge_state(current_state, updates)

            # Compute derived metrics
            derived = self._compute_derived_metrics(new_state)
            new_state['derived_metrics'] = derived

            # Increment version
            new_state['version'] = current_version + 1
            new_state['last_updated'] = pd.Timestamp.now().isoformat()
            new_state['last_update_source'] = source

            # Save to state store
            await self.state_store.save(asset_id, new_state)

            # Update cache
            self.cache[asset_id] = new_state

            # Publish state change event
            await self.event_bus.publish('twin.state.updated', {
                'asset_id': asset_id,
                'updates': updates,
                'new_state': new_state,
                'source': source
            })

            logger.info(f"Updated twin {asset_id} to version {new_state['version']}")

            return new_state

        async def get_twin_state(self, asset_id: str) -> Optional[Dict]:
            """Get current twin state (cached or from DB)"""
            # Check cache first
            if asset_id in self.cache:
                return self.cache[asset_id]

            # Load from state store
            state = await self.state_store.load(asset_id)

            if state:
                self.cache[asset_id] = state

            return state

        def _merge_state(
            self,
            current: Dict,
            updates: Dict
        ) -> Dict:
            """
            Merge updates into current state

            Strategy:
            - Sensor values: Replace
            - Arrays: Append
            - Nested objects: Deep merge
            """
            merged = current.copy()

            for key, value in updates.items():
                if key in ['expected_version', 'version', 'last_updated']:
                    # Skip metadata fields
                    continue

                if key not in merged:
                    # New field
                    merged[key] = value
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    # Deep merge for nested objects
                    merged[key] = self._merge_dict(merged[key], value)
                else:
                    # Replace
                    merged[key] = value

            return merged

        def _merge_dict(self, base: Dict, updates: Dict) -> Dict:
            """Deep merge dictionaries"""
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        def _compute_derived_metrics(self, state: Dict) -> Dict:
            """
            Compute derived metrics from sensor values

            Derived metrics:
            - Overall Equipment Effectiveness (OEE)
            - Efficiency
            - Health score (0-100)
            - Performance index
            """
            derived = {}

            # Health score (weighted combination of factors)
            health_score = self._compute_health_score(state)
            derived['health_score'] = health_score

            # Efficiency (power output / power input)
            if 'power_output' in state and 'power_input' in state:
                efficiency = state['power_output'] / state['power_input'] if state['power_input'] > 0 else 0
                derived['efficiency'] = round(efficiency, 3)

            # Performance index (actual vs. rated)
            if 'rpm' in state and 'rated_rpm' in state:
                performance = state['rpm'] / state['rated_rpm'] if state['rated_rpm'] > 0 else 0
                derived['performance_index'] = round(performance, 3)

            # Temperature margin (safety margin)
            if 'temperature' in state and 'max_temperature' in state:
                temp_margin = state['max_temperature'] - state['temperature']
                derived['temperature_margin'] = round(temp_margin, 1)

            return derived

        def _compute_health_score(self, state: Dict) -> int:
            """
            Compute overall health score (0-100)

            Factors:
            - Vibration level (30% weight)
            - Temperature (25% weight)
            - Anomaly frequency (20% weight)
            - Maintenance compliance (15% weight)
            - Operating hours vs. MTBF (10% weight)
            """
            score = 100

            # Vibration penalty
            if 'vibration' in state:
                vib = state['vibration']
                vib_normal = state.get('vibration_normal_max', 0.30)
                if vib > vib_normal:
                    penalty = min(30, (vib / vib_normal - 1) * 50)
                    score -= penalty

            # Temperature penalty
            if 'temperature' in state:
                temp = state['temperature']
                temp_normal = state.get('temp_normal_max', 70.0)
                if temp > temp_normal:
                    penalty = min(25, (temp / temp_normal - 1) * 50)
                    score -= penalty

            # Anomaly frequency penalty
            if 'anomaly_count_7d' in state:
                anomaly_count = state['anomaly_count_7d']
                if anomaly_count > 10:
                    penalty = min(20, (anomaly_count - 10) * 2)
                    score -= penalty

            # Maintenance compliance
            if 'hours_since_maintenance' in state and 'maintenance_interval' in state:
                hours_since = state['hours_since_maintenance']
                interval = state['maintenance_interval']
                if hours_since > interval:
                    penalty = min(15, (hours_since / interval - 1) * 30)
                    score -= penalty

            # Operating hours vs. MTBF
            if 'operating_hours' in state and 'mtbf' in state:
                hours = state['operating_hours']
                mtbf = state['mtbf']
                if hours > mtbf * 0.8:  # > 80% of MTBF
                    penalty = min(10, (hours / mtbf - 0.8) * 50)
                    score -= penalty

            return max(0, int(score))


    class TwinQueryAPI:
        """
        GraphQL API for querying digital twins

        Why GraphQL:
        - Flexible queries (clients specify what they need)
        - Type-safe schema
        - Real-time subscriptions (WebSocket)
        - Efficient (no over-fetching)
        """

        def __init__(self, twin_engine: DigitalTwinEngine):
            self.twin_engine = twin_engine
            self.schema = self._build_schema()

        def _build_schema(self):
            """Build GraphQL schema"""
            schema = """
                type Asset {
                    id: ID!
                    name: String!
                    type: String!
                    status: String!
                    healthScore: Int!
                    currentState: AssetState!
                    predictedFailure: FailurePrediction
                    maintenanceSchedule: MaintenanceSchedule
                    sensors: [Sensor!]!
                    children: [Asset!]
                }

                type AssetState {
                    temperature: Float
                    vibration: Float
                    rpm: Float
                    power: Float
                    efficiency: Float
                    lastUpdate: DateTime!
                }

                type FailurePrediction {
                    probability: Float!
                    timeToFailure: Int!
                    failureMode: String!
                    confidence: Float!
                }

                type MaintenanceSchedule {
                    nextScheduled: DateTime
                    lastCompleted: DateTime
                    workOrderId: String
                }

                type Sensor {
                    id: ID!
                    type: String!
                    unit: String!
                    currentValue: Float
                    status: String!
                }

                type Query {
                    asset(id: ID!): Asset
                    assets(filter: AssetFilter): [Asset!]!
                    assetsByHealth(minScore: Int, maxScore: Int): [Asset!]!
                }

                type Subscription {
                    assetStateChanged(assetId: ID!): Asset!
                    anomalyDetected(assetId: ID): Anomaly!
                }

                input AssetFilter {
                    type: String
                    factoryId: String
                    status: String
                }
            """
            return schema
    ```

    **Digital Twin Benefits:**

    ```
    1. Unified view:
       - Single source of truth for asset state
       - Combines sensor data, predictions, maintenance history
       - Accessible via API (no direct DB queries)

    2. Real-time updates:
       - WebSocket subscriptions for live data
       - Push updates to dashboards (no polling)
       - Reduced query load (95% reduction)

    3. Simulation:
       - "What-if" scenarios (e.g., delayed maintenance)
       - Optimize maintenance schedules
       - Training for operators

    4. Historical context:
       - Compare current vs. past states
       - Understand degradation trends
       - Learn from failures

    Performance:
    - State update latency: 50ms (edge to cloud)
    - Query latency: 10ms (cached), 100ms (DB)
    - Throughput: 100K updates/sec (distributed)
    - Storage: 1 MB per twin (state + metadata)
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Edge Layer (Distributed):
    - 5,000 edge gateways (20 machines each)
    - Independent operation (no coordination needed)
    - Scale by adding gateways (linear scaling)
    - Firmware updates via OTA (gradual rollout)

    Cloud Ingestion Layer (Stateless):
    - 50 ingestion nodes (5K readings/sec each)
    - Auto-scale based on Kafka lag
    - Each factory → dedicated Kafka partition
    - Scale to 100+ nodes if needed

    Time-Series Database (Sharded):
    - Shard by asset_id hash (consistent hashing)
    - Each shard: 2 TB data, 20M sensors
    - 10M sensors = 1 shard (with replication: 3 nodes)
    - Add shards as sensor count grows
    - Query routing based on asset_id

    ML Inference (Stateless):
    - 4 inference nodes (10 predictions/sec each)
    - Load models on startup (cache in memory)
    - Auto-scale based on request queue depth
    - Gradual model deployment (canary)

    ML Training (Batch):
    - 20 GPU nodes (500 models/week each)
    - Training jobs scheduled via Kubernetes
    - Spot instances for cost savings (70% cheaper)
    - Checkpoint/resume for preemption

    Digital Twin Service (Partitioned):
    - 10 twin nodes (10K twins each)
    - Partition by asset_id hash
    - Sticky sessions for WebSocket subscriptions
    - Scale based on twin count and query load
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Edge filtering** | 90% bandwidth reduction | Slight accuracy loss (aggregate vs. raw) |
    | **Edge ML inference** | 100ms → 13ms latency | Edge deployment complexity, model size limits |
    | **TimescaleDB compression** | 10:1 storage reduction | 5-10% CPU overhead, query latency +10% for compressed data |
    | **Continuous aggregations** | 100x faster queries | 20% storage overhead, eventual consistency (1-min lag) |
    | **Digital twin cache** | 10x faster queries | 50ms staleness, memory overhead (1 MB/twin) |
    | **Kafka partitioning** | Linear scaling | Rebalancing overhead, partition count limit |
    | **Feature store** | 50% faster training | Storage overhead (20 TB), feature drift risk |
    | **Model quantization** | 4x smaller models | 2-3% accuracy loss, suitable for edge deployment |

    ---

    ## Edge vs. Cloud Processing

    ```
    Edge Processing (Local Gateway):

    Advantages:
    - Ultra-low latency: < 100ms (vs. 500ms+ cloud)
    - Network resilience: Works during outages
    - Bandwidth savings: 90% reduction (1.58 Gbps vs. 15.8 Gbps)
    - Privacy: Sensitive data stays on-premises
    - Regulatory: Some regions require local processing

    Disadvantages:
    - Limited compute: CPU only, no GPUs
    - Model size limits: < 50 MB (vs. multi-GB cloud models)
    - Deployment complexity: 5,000 gateways to manage
    - Update coordination: Gradual OTA rollout
    - Storage limits: 24h buffer (vs. unlimited cloud)

    Decision matrix:

    Process at Edge:
    - Real-time anomaly detection (latency-critical)
    - Threshold violations (simple rules)
    - Data filtering/aggregation (reduce bandwidth)
    - Basic ML inference (small models < 50 MB)

    Process in Cloud:
    - Predictive maintenance (complex models, GPU)
    - Historical analysis (large time windows)
    - Model training (requires GPU cluster)
    - Cross-asset correlation (global view needed)
    - Long-term storage (5 years retention)
    ```

    ---

    ## Data Compression Strategies

    ```
    1. Time-series compression (TimescaleDB):
       - Columnar storage (similar values together)
       - Delta encoding (store differences, not absolute values)
       - Dictionary encoding (repeated strings)
       - Run-length encoding (long sequences of same value)
       - Result: 10:1 compression ratio
       - Example: 4.59 TB/day → 459 GB/day

    2. Protocol-level compression (gzip):
       - Compress HTTPS payloads (edge → cloud)
       - 2:1 compression ratio for JSON
       - CPU cost: 5% overhead (acceptable)
       - Example: 1.58 Gbps → 790 Mbps

    3. Aggregation (pre-processing):
       - Edge: 100 readings → 1 aggregate (100:1)
       - Only send aggregates for normal data
       - Send full-resolution for anomalies
       - Result: 90% overall data reduction

    4. Downsampling (time-based):
       - 15 min: Full resolution (90 days)
       - 1 min: 15x reduction (1 year)
       - 1 hour: 3600x reduction (5 years)
       - Result: 95% storage cost savings

    Combined effect:
    - Raw: 8.4 PB (5 years)
    - After edge filtering: 837.5 TB
    - After compression: 83.75 TB
    - After downsampling: 4.52 TB
    - Total reduction: 1,860x (99.95% savings)
    ```

    ---

    ## Multi-Site Federation

    ```
    Centralized Model (Chosen):

    Architecture:
    - 50 factories → single cloud platform
    - Edge gateways at each factory
    - Centralized data lake and ML training
    - Federated authentication (SSO)

    Advantages:
    - Unified view across all sites
    - Cross-site analytics (compare factories)
    - Centralized model training (more data = better models)
    - Consistent UI/UX
    - Lower operational overhead

    Challenges:
    - Network dependency (edge mitigates with local buffer)
    - Data sovereignty (some regions require local storage)
    - Latency for distant sites (edge processing helps)

    Hybrid approach for regulated regions:
    - Local data storage (compliance)
    - Replicate to central (analytics)
    - Federated learning (train locally, aggregate centrally)

    Cost:
    - Centralized: $50K/month (141 cloud nodes)
    - Distributed: $250K/month (5 regional clusters)
    - Savings: 80% cost reduction
    ```

    ---

    ## Failure Scenarios & Mitigation

    ```
    Scenario 1: Edge gateway failure
    - Impact: 20 machines lose connectivity
    - Mitigation: Redundant gateway (hot standby)
    - Recovery time: Automatic failover (< 1 minute)
    - Data loss: 0% (sensor data buffered in PLC)

    Scenario 2: Network outage (factory to cloud)
    - Impact: No cloud sync for affected factory
    - Mitigation: Edge continues processing, 24h buffer
    - Recovery time: Automatic reconnection
    - Data loss: 0% (buffer covers typical outages)

    Scenario 3: Time-series database node failure
    - Impact: Queries to affected shard fail
    - Mitigation: Replication (3x), automatic failover
    - Recovery time: < 30 seconds
    - Data loss: 0% (replicated)

    Scenario 4: ML inference service failure
    - Impact: No new RUL predictions
    - Mitigation: Multiple inference nodes, load balancer
    - Recovery time: Automatic (load balancer detects)
    - Data loss: 0% (predictions not critical real-time)

    Scenario 5: OPC UA server crash
    - Impact: Sensors can't send data
    - Mitigation: Server redundancy (primary + backup)
    - Recovery time: Automatic failover (< 10 seconds)
    - Data loss: < 10 seconds of data

    Scenario 6: Model prediction wrong (false positive)
    - Impact: Unnecessary maintenance scheduled
    - Mitigation: Human review for high-cost actions
    - Cost: Wasted maintenance ($2.5K) vs. missed failure ($250K)
    - Strategy: Err on side of caution (better safe than sorry)
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"Why use OPC UA instead of MQTT or REST?"**
   - OPC UA is the de facto standard for industrial IoT
   - Built-in security (X.509 certificates, encryption)
   - Vendor-neutral (works with all major PLCs)
   - Complex data types and discovery
   - Historical access (query past data)
   - **Trade-off:** Higher overhead than MQTT, but benefits outweigh cost

2. **"How do you handle network outages at the factory?"**
   - Edge gateways buffer 24 hours of data locally
   - Continue anomaly detection during outage
   - Alerts sent to local SCADA system
   - Automatic reconnection and sync when network restored
   - **Result:** 0% data loss for typical outages (< 24h)

3. **"Why process at edge instead of sending all data to cloud?"**
   - Latency: 100ms edge vs. 500ms+ cloud (critical for real-time)
   - Bandwidth: 90% reduction (1.58 Gbps vs. 15.8 Gbps)
   - Cost: $500K/year bandwidth savings
   - Resilience: Works during cloud/network outages
   - **Trade-off:** Edge deployment complexity

4. **"How does survival analysis differ from classification for predictive maintenance?"**
   - Classification: Binary prediction (will fail: yes/no)
   - Survival analysis: Time-to-failure prediction (when will it fail)
   - Handles censored data (equipment still running)
   - Provides confidence intervals (uncertainty)
   - **Result:** More actionable predictions for maintenance planning

5. **"What's the difference between digital twin and device shadow?"**
   - Device shadow: Simple key-value state (IoT Core concept)
   - Digital twin: Rich model with history, predictions, physics
   - Twin includes: Current state + ML predictions + maintenance history
   - Twin supports simulation ("what-if" scenarios)
   - **Use case:** Digital twin for complex assets, shadow for simple devices

6. **"How do you ensure ML model accuracy over time?"**
   - Monitor prediction accuracy (actual failures vs. predicted)
   - Retrain weekly with new failure data
   - A/B testing (new model on 10% of fleet first)
   - Feature drift detection (alert if data distribution changes)
   - Human feedback loop (operators flag false positives)
   - **Target:** Maintain > 85% accuracy

7. **"How do you scale to 1M machines instead of 100K?"**
   - Edge layer: Linear scaling (add more gateways)
   - Cloud ingestion: Auto-scale based on Kafka lag
   - Time-series DB: Add shards (10x more shards)
   - ML training: More GPU nodes (train in parallel)
   - Cost scaling: Near-linear (edge dominates cost)
   - **Result:** Architecture supports 10x scale with no redesign

**Key Points to Mention:**

- Edge-first architecture (90% data reduction, < 100ms latency)
- OPC UA protocol for industrial standard compliance
- Multi-level anomaly detection (rule + statistical + ML)
- Survival analysis for RUL prediction (not just classification)
- Digital twin for unified asset view
- TimescaleDB for time-series optimization (10:1 compression)
- Multi-site federation with centralized cloud platform
- Network resilience via edge buffering (24h)

---

## Real-World Examples

**GE Predix:**
- Industrial IoT platform for asset performance management
- Used by airlines (jet engine monitoring), utilities (wind turbines)
- Edge analytics with cloud-based AI
- Predictive maintenance reduces unplanned downtime by 20%
- Digital twin for asset optimization
- Architecture: Edge gateways → Predix cloud → Analytical apps

**Siemens MindSphere:**
- Open IoT operating system for connected products
- Used by manufacturers (automotive, electronics)
- OPC UA and industrial protocol support
- Fleet management and asset monitoring
- AI/ML for anomaly detection and RUL prediction
- Integration with Siemens PLCs and SCADA

**AWS IoT for Industrial:**
- SiteWise: Industrial data collection and visualization
- TwinMaker: Digital twin service
- Greengrass: Edge computing runtime
- Used by BP (oil & gas), Volkswagen (manufacturing)
- Predictive maintenance reduces maintenance costs by 30%

**Honeywell Connected Plant:**
- Industrial IoT solution for process industries
- Real-time monitoring and optimization
- Predictive analytics for equipment health
- Integration with Honeywell DCS and safety systems
- Cybersecurity for OT environments

---

## Summary

**System Characteristics:**

- **Scale:** 100K machines, 10M sensors, 1B data points/day
- **Latency:** < 100ms edge, < 2s cloud queries
- **Storage:** 38.8 TB total (with compression + edge filtering)
- **Availability:** 99.9% uptime
- **Prediction Accuracy:** > 85% for failure prediction

**Core Components:**

1. **Edge Gateways:** OPC UA collection, real-time anomaly detection, 24h buffer
2. **Time-Series Database:** TimescaleDB with 10:1 compression, multi-tier storage
3. **Digital Twin Engine:** Real-time state synchronization, GraphQL API
4. **Predictive Maintenance:** Survival analysis (Cox PH), RUL prediction
5. **Alert Manager:** Multi-level detection, rule engine, technician assignment
6. **Maintenance Scheduler:** Automated work orders, CMMS integration
7. **ML Training Pipeline:** Weekly retraining, GPU cluster, gradual deployment

**Key Design Decisions:**

- Edge-first architecture (90% bandwidth savings, < 100ms latency)
- OPC UA protocol (industrial standard, secure, vendor-neutral)
- Survival analysis for RUL (not just binary classification)
- Digital twin pattern (unified view, simulation capability)
- Multi-level anomaly detection (rule + statistical + ML)
- TimescaleDB for time-series (10:1 compression, continuous aggregations)
- Multi-site federation (centralized cloud, edge resilience)
- 24-hour edge buffering (network resilience)

This design provides a scalable, resilient Industrial IoT platform capable of monitoring millions of sensors in real-time, predicting equipment failures days in advance, and optimizing maintenance schedules to minimize downtime and costs. The edge-first architecture ensures low latency and network resilience, while the cloud platform enables advanced ML training and cross-site analytics.
