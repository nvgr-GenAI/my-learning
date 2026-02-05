# Design Smart Meter System (Electricity Consumption)

A real-time electricity consumption monitoring and analytics platform that collects meter readings from millions of smart meters, provides usage insights, calculates bills, and detects anomalies at scale.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M meters, 96 readings/day per meter, real-time analytics, 15-minute intervals |
| **Key Challenges** | Time-series data ingestion, data aggregation, billing accuracy, anomaly detection, network reliability |
| **Core Concepts** | Time-series database, data collectors, downsampling, time-of-use pricing, demand response |
| **Companies** | Sense, Emporia, Itron, Landis+Gyr, Enel, Pacific Gas & Electric (PG&E), National Grid |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Real-time Consumption** | Track electricity usage in 15-minute intervals | P0 (Must have) |
    | **Usage Analytics** | Daily/monthly consumption patterns and insights | P0 (Must have) |
    | **Billing Calculation** | Calculate bills with time-of-use rates | P0 (Must have) |
    | **Historical Data** | Access historical consumption data | P0 (Must have) |
    | **Anomaly Detection** | Detect unusual consumption, power outages | P0 (Must have) |
    | **Mobile App** | Real-time monitoring and notifications | P1 (Should have) |
    | **Demand Response** | Load balancing and peak demand management | P1 (Should have) |
    | **Appliance Breakdown** | Identify individual appliance consumption | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Solar panel integration (net metering)
    - Electric vehicle charging management
    - Home automation and IoT device control
    - Energy trading/marketplace
    - Detailed power quality metrics (voltage, harmonics)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.95% uptime | Critical for billing and grid management |
    | **Data Loss** | < 0.1% of readings | Billing accuracy requires reliable data |
    | **Ingestion Latency** | < 5 minutes | Near real-time monitoring for users |
    | **Query Latency** | < 2s for dashboards | Fast user experience |
    | **Data Retention** | 3 years full data | Regulatory compliance and analytics |
    | **Scalability** | 10M+ meters | Support utility-scale deployments |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total smart meters: 10,000,000 meters
    Reading interval: 15 minutes (96 readings/day)
    Data per reading: Energy (kWh), voltage, current, power factor

    Daily readings:
    - Total: 10M × 96 = 960M readings/day
    - Per second: 960M / 86,400s = 11,111 readings/sec
    - Peak (network congestion, retry bursts): 33,333 readings/sec

    Real-time queries (mobile app):
    - Active users: 2M daily active (20% of users)
    - Queries per user: 10 queries/day
    - Total: 2M × 10 = 20M queries/day = 231 queries/sec
    - Peak: 700 queries/sec (evening hours)

    Billing calculations:
    - Monthly billing cycles: 10M bills/month
    - Calculation window: 5 days
    - Rate: 10M / (5 × 86,400s) = 23 bills/sec

    Anomaly detection:
    - Continuous evaluation on all readings
    - Processing: 11,111 readings/sec
    - Alert generation: ~100 alerts/sec (1% anomaly rate)
    ```

    ### Storage Estimates

    ```
    Per reading data point:
    - Meter ID: 8 bytes (UUID)
    - Timestamp: 8 bytes (Unix timestamp)
    - Energy consumed: 4 bytes (float32, kWh)
    - Power: 4 bytes (float32, kW)
    - Voltage: 2 bytes (int16, volts)
    - Current: 2 bytes (int16, amperes)
    - Power factor: 2 bytes (int16, 0-100)
    - Status flags: 1 byte (outage, tamper, etc.)
    - Total per reading: 31 bytes

    Raw storage (15-minute resolution):
    - Daily: 960M readings × 31 bytes = 29.8 GB/day
    - Yearly: 29.8 GB × 365 = 10.87 TB/year
    - 3 years: 32.6 TB

    With compression (8:1 ratio for time-series):
    - 3 years: 32.6 TB / 8 = 4.1 TB compressed

    Retention strategy (multi-tier):

    Tier 1: Full resolution (15 min) - 90 days
    - Storage: 29.8 GB/day × 90 = 2.68 TB
    - Compressed: 335 GB
    - Use: Real-time monitoring, billing

    Tier 2: Hourly aggregation - 1 year
    - Reduction: 4x fewer points
    - Storage: 10.87 TB / 4 = 2.72 TB
    - Compressed: 340 GB
    - Use: Monthly reports, trend analysis

    Tier 3: Daily aggregation - 3 years
    - Reduction: 96x fewer points
    - Storage: 10.87 TB × 3 / 96 = 340 GB
    - Compressed: 42 GB
    - Use: Historical comparisons, long-term trends

    Total storage: 335 GB + 340 GB + 42 GB = 717 GB
    With replication (3x): 2.15 TB

    User data:
    - 10M users × 2 KB (profile, address, rate plan) = 20 GB

    Billing data:
    - 10M bills/month × 5 KB = 50 GB/month
    - 3 years: 50 GB × 36 = 1.8 TB

    Total: 2.15 TB (readings) + 20 GB (users) + 1.8 TB (bills) = 4 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (meter readings):
    - 11,111 readings/sec × 31 bytes = 344 KB/sec ≈ 2.75 Mbps
    - Compressed (2:1): 1.4 Mbps
    - With protocol overhead (HTTP, TLS): 2.8 Mbps
    - Peak (3x): 8.4 Mbps

    Egress (queries, dashboards):
    - 231 queries/sec × 1,000 readings × 31 bytes = 7.2 MB/sec ≈ 57 Mbps
    - Dashboard updates: 500K active dashboards × 100 KB / 300s = 167 MB/sec
    - Mobile apps: 100K concurrent × 10 KB/sec = 1 GB/sec
    - Total egress: ~1.2 Gbps

    Total bandwidth: 10 Mbps (ingress) + 1.2 Gbps (egress) = 1.21 Gbps
    ```

    ### Server Estimates

    ```
    Data collection layer:
    - 11,111 readings/sec / 1,000 per node = 12 collector nodes
    - CPU: 2 cores per node (protocol handling)
    - Memory: 8 GB per node (buffering)

    Time-series database:
    - 4 TB total / 2 TB per node = 2 TSDB nodes
    - CPU: 16 cores per node (compression, indexing)
    - Memory: 128 GB per node (hot data, indexes)
    - Disk: 2 TB SSD per node

    Query/API layer:
    - 700 queries/sec / 100 per node = 7 API nodes
    - CPU: 8 cores per node (aggregation)
    - Memory: 32 GB per node (query cache)

    Analytics/anomaly detection:
    - 11,111 readings/sec / 2,000 per node = 6 analytics nodes
    - CPU: 8 cores per node (ML models)
    - Memory: 32 GB per node (model state)

    Total servers:
    - Collection: 12 nodes
    - Storage: 2 nodes (with replication: 6 nodes)
    - Query: 7 nodes
    - Analytics: 6 nodes
    - Total: ~31 nodes
    ```

    ---

    ## Key Assumptions

    1. 15-minute reading interval (standard for smart meters)
    2. 96 readings per meter per day
    3. 20% of users check app daily (2M active users)
    4. 8:1 compression ratio for time-series data
    5. 99% of queries access last 7 days of data
    6. Network connectivity: 95% success rate (5% retry/buffering)
    7. Average household consumption: 30 kWh/day (900 kWh/month)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Time-series optimization** - Efficient storage and querying of meter readings
    2. **Buffering and retry** - Handle intermittent network connectivity
    3. **Data aggregation** - Pre-compute hourly/daily summaries
    4. **Event-driven processing** - Asynchronous anomaly detection and alerts
    5. **Multi-tier storage** - Balance cost and query performance

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Smart Meters"
            Meter1[Smart Meter 1<br/>15-min readings]
            Meter2[Smart Meter 2<br/>15-min readings]
            MeterN[Smart Meter N<br/>15-min readings]
        end

        subgraph "Data Collection Layer"
            Collector1[Data Collector 1<br/>Cellular/WiFi Gateway]
            Collector2[Data Collector 2<br/>Cellular/WiFi Gateway]
            CollectorN[Data Collector N<br/>Cellular/WiFi Gateway]
            Buffer[Message Buffer<br/>Kafka<br/>Retry queue]
        end

        subgraph "Ingestion Pipeline"
            Validator[Data Validator<br/>Sanity checks<br/>Deduplication]
            Enricher[Data Enricher<br/>Add metadata<br/>Rate plans]
            BatchWriter[Batch Writer<br/>Batch to TSDB]
        end

        subgraph "Time-Series Storage"
            TSDB_Hot[(Hot Storage<br/>InfluxDB/TimescaleDB<br/>90 days<br/>15-min resolution)]
            TSDB_Warm[(Warm Storage<br/>TimescaleDB<br/>1 year<br/>1-hour resolution)]
            TSDB_Cold[(Cold Storage<br/>S3/Parquet<br/>3 years<br/>1-day resolution)]
        end

        subgraph "Aggregation & Processing"
            Aggregator[Aggregation Service<br/>Hourly/daily rollups]
            AnomalyDetector[Anomaly Detector<br/>ML models<br/>Pattern detection]
            BillingEngine[Billing Engine<br/>TOU rates<br/>Calculations]
        end

        subgraph "Analytics & Query"
            QueryEngine[Query Engine<br/>Time-series queries]
            Cache[Query Cache<br/>Redis]
            Analytics[Analytics Service<br/>Usage patterns<br/>Insights]
        end

        subgraph "API & Application"
            API_Gateway[API Gateway<br/>REST/GraphQL<br/>Auth]
            MobileApp[Mobile App<br/>Real-time dashboard]
            WebPortal[Web Portal<br/>Billing, reports]
            AdminPanel[Admin Panel<br/>Operations]
        end

        subgraph "Notification & Alerts"
            AlertManager[Alert Manager<br/>Rules engine]
            NotificationService[Notification Service<br/>Push, SMS, Email]
        end

        subgraph "External Integrations"
            WeatherAPI[Weather API<br/>Temperature data]
            BillingSystem[Billing System<br/>Invoice generation]
            GridOperator[Grid Operator<br/>Demand response]
        end

        Meter1 -.->|Cellular/WiFi| Collector1
        Meter2 -.->|Cellular/WiFi| Collector1
        MeterN -.->|Cellular/WiFi| CollectorN

        Collector1 --> Buffer
        Collector2 --> Buffer
        CollectorN --> Buffer

        Buffer --> Validator
        Validator --> Enricher
        Enricher --> BatchWriter

        BatchWriter --> TSDB_Hot

        TSDB_Hot --> Aggregator
        Aggregator --> TSDB_Warm
        Aggregator --> TSDB_Cold

        TSDB_Hot --> AnomalyDetector
        AnomalyDetector --> AlertManager
        AlertManager --> NotificationService

        TSDB_Hot --> BillingEngine
        TSDB_Warm --> BillingEngine
        BillingEngine --> BillingSystem

        API_Gateway --> QueryEngine
        QueryEngine --> Cache
        QueryEngine --> TSDB_Hot
        QueryEngine --> TSDB_Warm
        QueryEngine --> TSDB_Cold
        QueryEngine --> Analytics

        MobileApp --> API_Gateway
        WebPortal --> API_Gateway
        AdminPanel --> API_Gateway

        NotificationService --> MobileApp

        WeatherAPI --> Analytics
        BillingEngine --> GridOperator

        style Buffer fill:#e8eaf6
        style TSDB_Hot fill:#ffe1e1
        style TSDB_Warm fill:#fff4e1
        style TSDB_Cold fill:#f0f0f0
        style Cache fill:#fff4e1
        style API_Gateway fill:#e1f5ff
        style AnomalyDetector fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Kafka Message Buffer** | Handle network failures, retry failed transmissions, decouple collectors from processing | Direct writes (data loss on failure), database queue (too slow) |
    | **TimescaleDB** | PostgreSQL extension for time-series, SQL queries, compression, continuous aggregations | InfluxDB (less mature SQL support), Cassandra (no built-in aggregations), MongoDB (poor compression) |
    | **Multi-tier Storage** | 95% cost savings vs. keeping all data hot | Single-tier SSD (expensive for 3 years), All on S3 (slow queries) |
    | **Batch Writer** | Reduce database write load, group small writes | Individual writes (overload TSDB), too large batches (high latency) |
    | **Redis Cache** | 80% cache hit rate, 10x faster queries | No cache (slow repeated queries), application cache (no sharing) |
    | **Continuous Aggregations** | Pre-compute hourly/daily summaries (100x faster) | On-demand aggregation (too slow for dashboards) |

    **Key Trade-off:** We chose **eventual consistency** for meter readings (acceptable delay: 5 minutes) to handle network failures gracefully. Strong consistency would result in data loss during outages.

    ---

    ## API Design

    ### 1. Submit Meter Reading (Meter to Collector)

    **Request:**
    ```http
    POST /api/v1/readings
    Content-Type: application/json
    Authorization: Bearer <meter_token>

    {
      "meter_id": "meter_a1b2c3d4",
      "readings": [
        {
          "timestamp": 1735819200,
          "energy_kwh": 0.65,        // Energy consumed in 15 min
          "power_kw": 2.6,           // Average power
          "voltage_v": 120,
          "current_a": 21.7,
          "power_factor": 95,
          "status": "normal"          // normal, outage, tamper
        },
        {
          "timestamp": 1735820100,
          "energy_kwh": 0.72,
          "power_kw": 2.88,
          "voltage_v": 119,
          "current_a": 24.2,
          "power_factor": 94,
          "status": "normal"
        }
      ],
      "device_info": {
        "firmware_version": "2.3.1",
        "signal_strength": -65       // dBm
      }
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "accepted": 2,
      "rejected": 0,
      "next_upload": 1735820200
    }
    ```

    **Design Notes:**

    - Batch multiple readings in single request (reduce network overhead)
    - Include device diagnostics for monitoring
    - Return next upload time for adaptive intervals
    - Authenticate with meter-specific token (certificate-based)

    ---

    ### 2. Get Current Consumption

    **Request:**
    ```http
    GET /api/v1/meters/meter_a1b2c3d4/consumption/current
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "meter_id": "meter_a1b2c3d4",
      "current_power_kw": 2.8,
      "today_energy_kwh": 18.5,
      "today_cost_usd": 3.25,
      "last_reading": {
        "timestamp": 1735820100,
        "energy_kwh": 0.72,
        "power_kw": 2.88,
        "voltage_v": 119
      },
      "status": "normal",
      "comparison": {
        "yesterday_same_time": 15.2,
        "last_week_same_day": 17.8,
        "monthly_average": 16.5
      }
    }
    ```

    **Design Notes:**

    - Return current power draw (latest reading)
    - Include today's cumulative consumption
    - Provide comparisons for context
    - Cache this endpoint (30-second TTL)

    ---

    ### 3. Get Usage History

    **Request:**
    ```http
    GET /api/v1/meters/meter_a1b2c3d4/usage/history?start=2025-01-01&end=2025-01-31&resolution=daily
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "meter_id": "meter_a1b2c3d4",
      "resolution": "daily",
      "data": [
        {
          "date": "2025-01-01",
          "energy_kwh": 28.5,
          "cost_usd": 5.10,
          "peak_power_kw": 4.2,
          "breakdown": {
            "peak_hours": 8.5,      // kWh during peak hours
            "off_peak_hours": 20.0
          }
        },
        {
          "date": "2025-01-02",
          "energy_kwh": 32.1,
          "cost_usd": 5.85,
          "peak_power_kw": 5.1,
          "breakdown": {
            "peak_hours": 12.3,
            "off_peak_hours": 19.8
          }
        }
        // ... more days
      ],
      "summary": {
        "total_energy_kwh": 875.5,
        "total_cost_usd": 157.50,
        "average_daily_kwh": 28.2,
        "peak_demand_kw": 6.8
      }
    }
    ```

    **Query Parameters:**

    - `resolution`: 15min, hourly, daily, monthly
    - `start`, `end`: Date range
    - `include_breakdown`: Include peak/off-peak split

    ---

    ### 4. Calculate Bill

    **Request:**
    ```http
    POST /api/v1/billing/calculate
    Content-Type: application/json
    Authorization: Bearer <user_token>

    {
      "meter_id": "meter_a1b2c3d4",
      "billing_period": {
        "start": "2025-01-01",
        "end": "2025-01-31"
      },
      "rate_plan": "residential_tou"
    }
    ```

    **Response:**
    ```json
    {
      "bill_id": "bill_xyz123",
      "meter_id": "meter_a1b2c3d4",
      "billing_period": {
        "start": "2025-01-01",
        "end": "2025-01-31"
      },
      "consumption": {
        "total_kwh": 875.5,
        "peak_kwh": 285.3,
        "off_peak_kwh": 590.2
      },
      "charges": {
        "energy_charge": {
          "peak_hours": {
            "kwh": 285.3,
            "rate": 0.28,
            "amount": 79.88
          },
          "off_peak_hours": {
            "kwh": 590.2,
            "rate": 0.12,
            "amount": 70.82
          }
        },
        "demand_charge": {
          "peak_demand_kw": 6.8,
          "rate": 12.50,
          "amount": 85.00
        },
        "fixed_charge": 15.00,
        "taxes": 25.07,
        "total": 275.77
      },
      "rate_plan": {
        "name": "Residential Time-of-Use",
        "peak_hours": "14:00-20:00 weekdays",
        "peak_rate": 0.28,
        "off_peak_rate": 0.12
      }
    }
    ```

    **Design Notes:**

    - Support multiple rate structures (flat, TOU, tiered)
    - Include demand charges (commercial customers)
    - Show detailed breakdown for transparency
    - Cache calculation (bills don't change once generated)

    ---

    ### 5. Report Anomaly (Internal API)

    **Request:**
    ```http
    POST /internal/api/v1/anomalies
    Content-Type: application/json

    {
      "meter_id": "meter_a1b2c3d4",
      "anomaly_type": "spike",      // spike, drop, outage, tamper
      "severity": "high",
      "detected_at": 1735820100,
      "details": {
        "expected_power_kw": 2.5,
        "actual_power_kw": 8.2,
        "deviation_percent": 228,
        "confidence": 0.95
      },
      "context": {
        "temperature_f": 85,
        "day_of_week": "wednesday",
        "time_of_day": "15:30"
      }
    }
    ```

    **Response:**
    ```json
    {
      "anomaly_id": "anom_abc789",
      "alert_sent": true,
      "action": "notify_user"
    }
    ```

    ---

    ## Database Schema

    ### Time-Series Data (TimescaleDB)

    **Meter Readings (Hypertable):**

    ```sql
    -- Create hypertable for automatic time-based partitioning
    CREATE TABLE meter_readings (
        meter_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        energy_kwh REAL NOT NULL,           -- Energy consumed in interval
        power_kw REAL NOT NULL,             -- Average power
        voltage_v SMALLINT,
        current_a REAL,
        power_factor SMALLINT,              -- 0-100
        status VARCHAR(20) DEFAULT 'normal',
        PRIMARY KEY (meter_id, time)
    );

    -- Convert to hypertable (automatic partitioning by time)
    SELECT create_hypertable('meter_readings', 'time');

    -- Enable compression (8:1 ratio)
    ALTER TABLE meter_readings SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'meter_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    -- Compression policy: compress data older than 7 days
    SELECT add_compression_policy('meter_readings', INTERVAL '7 days');

    -- Retention policy: drop data older than 3 years
    SELECT add_retention_policy('meter_readings', INTERVAL '3 years');

    -- Indexes
    CREATE INDEX idx_meter_time ON meter_readings (meter_id, time DESC);
    CREATE INDEX idx_time ON meter_readings (time DESC);
    ```

    **Continuous Aggregations (Pre-computed Summaries):**

    ```sql
    -- Hourly aggregation (refreshed every 30 minutes)
    CREATE MATERIALIZED VIEW meter_readings_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        meter_id,
        time_bucket('1 hour', time) AS bucket,
        SUM(energy_kwh) as total_energy_kwh,
        AVG(power_kw) as avg_power_kw,
        MAX(power_kw) as peak_power_kw,
        MIN(power_kw) as min_power_kw,
        AVG(voltage_v) as avg_voltage_v,
        COUNT(*) as reading_count
    FROM meter_readings
    GROUP BY meter_id, bucket;

    SELECT add_continuous_aggregate_policy('meter_readings_hourly',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '30 minutes'
    );

    -- Daily aggregation
    CREATE MATERIALIZED VIEW meter_readings_daily
    WITH (timescaledb.continuous) AS
    SELECT
        meter_id,
        time_bucket('1 day', time) AS bucket,
        SUM(energy_kwh) as total_energy_kwh,
        AVG(power_kw) as avg_power_kw,
        MAX(power_kw) as peak_power_kw,
        MIN(voltage_v) as min_voltage_v,
        MAX(voltage_v) as max_voltage_v,
        COUNT(*) as reading_count
    FROM meter_readings
    GROUP BY meter_id, bucket;

    SELECT add_continuous_aggregate_policy('meter_readings_daily',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day'
    );
    ```

    **Why Continuous Aggregations:**

    - Pre-compute hourly/daily summaries incrementally
    - 100x faster queries for dashboards
    - Automatic refresh as new data arrives
    - Reduce query load on raw data

    ---

    ### User & Billing Data (PostgreSQL)

    **Users:**

    ```sql
    CREATE TABLE users (
        user_id UUID PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        phone VARCHAR(20),
        full_name VARCHAR(255),
        address TEXT,
        rate_plan_id UUID REFERENCES rate_plans(rate_plan_id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX idx_user_email ON users(email);
    ```

    **Meters:**

    ```sql
    CREATE TABLE meters (
        meter_id UUID PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        meter_number VARCHAR(50) UNIQUE NOT NULL,
        installation_date DATE,
        location GEOGRAPHY(POINT),          -- PostGIS for location
        status VARCHAR(20) DEFAULT 'active', -- active, inactive, maintenance
        firmware_version VARCHAR(20),
        last_reading_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_meters (user_id),
        INDEX idx_status (status)
    );
    ```

    **Rate Plans:**

    ```sql
    CREATE TABLE rate_plans (
        rate_plan_id UUID PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        type VARCHAR(20) NOT NULL,          -- flat, tou, tiered
        description TEXT,
        rates JSONB NOT NULL,               -- Flexible rate structure
        effective_date DATE NOT NULL,
        expiry_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Example TOU rate plan:
    INSERT INTO rate_plans (rate_plan_id, name, type, rates) VALUES (
        'rp_tou_001',
        'Residential Time-of-Use',
        'tou',
        '{
            "peak": {
                "rate": 0.28,
                "hours": "14:00-20:00",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "off_peak": {
                "rate": 0.12,
                "hours": "00:00-14:00,20:00-24:00",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            },
            "demand_charge": 12.50,
            "fixed_charge": 15.00
        }'
    );
    ```

    **Bills:**

    ```sql
    CREATE TABLE bills (
        bill_id UUID PRIMARY KEY,
        meter_id UUID REFERENCES meters(meter_id),
        user_id UUID REFERENCES users(user_id),
        billing_period_start DATE NOT NULL,
        billing_period_end DATE NOT NULL,
        total_kwh DECIMAL(10, 2),
        peak_kwh DECIMAL(10, 2),
        off_peak_kwh DECIMAL(10, 2),
        peak_demand_kw DECIMAL(8, 2),
        total_amount DECIMAL(10, 2),
        status VARCHAR(20) DEFAULT 'pending',  -- pending, paid, overdue
        due_date DATE,
        paid_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_meter_period (meter_id, billing_period_start),
        INDEX idx_user_bills (user_id, created_at DESC),
        INDEX idx_status (status)
    );
    ```

    **Anomalies:**

    ```sql
    CREATE TABLE anomalies (
        anomaly_id UUID PRIMARY KEY,
        meter_id UUID REFERENCES meters(meter_id),
        anomaly_type VARCHAR(20) NOT NULL,  -- spike, drop, outage, tamper
        severity VARCHAR(10) NOT NULL,      -- low, medium, high, critical
        detected_at TIMESTAMP NOT NULL,
        resolved_at TIMESTAMP,
        expected_value REAL,
        actual_value REAL,
        confidence REAL,                    -- ML model confidence (0-1)
        details JSONB,
        status VARCHAR(20) DEFAULT 'open',  -- open, investigating, resolved, false_positive
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_meter_anomalies (meter_id, detected_at DESC),
        INDEX idx_status (status, detected_at DESC)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Meter Reading Collection Flow

    ```mermaid
    sequenceDiagram
        participant Meter as Smart Meter
        participant Collector as Data Collector
        participant Kafka
        participant Validator
        participant Enricher
        participant BatchWriter
        participant TSDB as TimescaleDB

        Note over Meter: Every 15 minutes

        Meter->>Meter: Measure consumption<br/>Buffer readings
        Meter->>Collector: HTTPS POST /readings<br/>(batch of 4 readings)

        alt Network available
            Collector->>Kafka: Publish to readings topic
            Kafka-->>Collector: Ack
            Collector-->>Meter: 200 OK, next_upload
        else Network unavailable
            Meter->>Meter: Buffer in local storage
            Note over Meter: Retry in 5 minutes<br/>(exponential backoff)
        end

        Kafka->>Validator: Consume batch
        Validator->>Validator: Check timestamp<br/>Validate ranges<br/>Deduplicate

        alt Valid readings
            Validator->>Enricher: Forward readings
            Enricher->>Enricher: Add user metadata<br/>Add rate plan<br/>Calculate cost
            Enricher->>BatchWriter: Enrich readings

            BatchWriter->>BatchWriter: Buffer 1000 readings<br/>or 10 seconds
            BatchWriter->>TSDB: Batch INSERT
            TSDB-->>BatchWriter: Success
        else Invalid readings
            Validator->>Validator: Log error<br/>Send alert
        end

        Note over TSDB: Continuous aggregation<br/>refreshes hourly view
    ```

    ---

    ### Billing Calculation Flow

    ```mermaid
    sequenceDiagram
        participant Scheduler
        participant BillingEngine
        participant TSDB as TimescaleDB
        participant RatePlan as Rate Plans DB
        participant BillDB as Bills DB
        participant Email as Email Service

        Note over Scheduler: Monthly billing cycle<br/>(1st of month)

        Scheduler->>BillingEngine: Start billing run<br/>for all meters

        loop For each meter
            BillingEngine->>TSDB: Query consumption<br/>SELECT * FROM meter_readings_hourly<br/>WHERE time BETWEEN start AND end
            TSDB-->>BillingEngine: Hourly readings

            BillingEngine->>RatePlan: Get rate plan for user
            RatePlan-->>BillingEngine: TOU rates, demand charges

            BillingEngine->>BillingEngine: Calculate charges:<br/>1. Split peak/off-peak<br/>2. Apply TOU rates<br/>3. Add demand charge<br/>4. Add fixed charge<br/>5. Calculate taxes

            BillingEngine->>BillDB: INSERT bill
            BillDB-->>BillingEngine: bill_id

            BillingEngine->>Email: Send bill notification
        end

        BillingEngine->>Scheduler: Billing run complete<br/>10M bills generated
    ```

    ---

    ### Anomaly Detection Flow

    ```mermaid
    sequenceDiagram
        participant TSDB as TimescaleDB
        participant AnomalyDetector
        participant MLModel as ML Model
        participant WeatherAPI
        participant AlertManager
        participant User as Mobile App

        Note over TSDB: New readings arrive

        TSDB->>AnomalyDetector: Stream new readings<br/>(Kafka consumer)

        AnomalyDetector->>TSDB: Get historical baseline<br/>(same hour, last 30 days)
        TSDB-->>AnomalyDetector: Historical data

        AnomalyDetector->>WeatherAPI: Get current temperature
        WeatherAPI-->>AnomalyDetector: Temp: 85°F

        AnomalyDetector->>MLModel: Predict expected power<br/>Input: time, day, temp, historical
        MLModel-->>AnomalyDetector: Expected: 2.5 kW<br/>Confidence: 0.95

        AnomalyDetector->>AnomalyDetector: Compare actual vs expected<br/>Actual: 8.2 kW<br/>Deviation: 228%

        alt Anomaly detected (>3 std deviations)
            AnomalyDetector->>AlertManager: Create anomaly alert<br/>Type: spike, Severity: high
            AlertManager->>AlertManager: Check alert rules<br/>Deduplicate
            AlertManager->>User: Push notification<br/>"Unusual spike in power usage"
            User-->>AlertManager: Ack notification
        else Normal usage
            AnomalyDetector->>AnomalyDetector: Update baseline model
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Time-Series Database Optimization

    ### TimescaleDB Compression & Partitioning

    ```python
    class MeterReadingStorage:
        """
        Efficient time-series storage for smart meter readings

        Key optimizations:
        1. Automatic time-based partitioning (hypertables)
        2. Columnar compression (8:1 ratio)
        3. Continuous aggregations (pre-computed summaries)
        4. Automatic retention policies
        """

        def __init__(self, db_connection):
            self.db = db_connection
            self.batch_size = 1000
            self.batch_buffer = []
            self.last_flush = time.time()

        def insert_reading(self, reading):
            """
            Buffer readings for batch insert

            Batching reduces write load by 100x:
            - Individual inserts: 1000 writes/sec max
            - Batch inserts: 100,000 writes/sec
            """
            self.batch_buffer.append(reading)

            # Flush if buffer full or timeout
            if len(self.batch_buffer) >= self.batch_size or \
               time.time() - self.last_flush > 10:
                self.flush_batch()

        def flush_batch(self):
            """Batch insert with COPY command (10x faster than INSERT)"""
            if not self.batch_buffer:
                return

            # Use PostgreSQL COPY for bulk insert
            # COPY is 10x faster than individual INSERTs
            csv_data = self._to_csv(self.batch_buffer)

            cursor = self.db.cursor()
            cursor.copy_expert(
                """
                COPY meter_readings
                (meter_id, time, energy_kwh, power_kw, voltage_v,
                 current_a, power_factor, status)
                FROM STDIN WITH CSV
                """,
                csv_data
            )

            self.db.commit()

            logger.info(f"Inserted {len(self.batch_buffer)} readings")

            self.batch_buffer.clear()
            self.last_flush = time.time()

        def query_consumption(self, meter_id, start_time, end_time, resolution='15min'):
            """
            Query consumption with automatic aggregation level selection

            Optimization: Use pre-computed aggregations when possible
            - 15min resolution: Query raw data
            - Hourly resolution: Query hourly continuous aggregate (4x faster)
            - Daily resolution: Query daily continuous aggregate (96x faster)
            """
            time_range_hours = (end_time - start_time).total_seconds() / 3600

            if resolution == 'daily' or time_range_hours > 168:  # > 1 week
                # Use daily aggregation
                query = """
                    SELECT
                        bucket as time,
                        total_energy_kwh,
                        avg_power_kw,
                        peak_power_kw
                    FROM meter_readings_daily
                    WHERE meter_id = %s
                      AND bucket BETWEEN %s AND %s
                    ORDER BY bucket
                """
                table = 'daily'

            elif resolution == 'hourly' or time_range_hours > 24:  # > 1 day
                # Use hourly aggregation
                query = """
                    SELECT
                        bucket as time,
                        total_energy_kwh,
                        avg_power_kw,
                        peak_power_kw
                    FROM meter_readings_hourly
                    WHERE meter_id = %s
                      AND bucket BETWEEN %s AND %s
                    ORDER BY bucket
                """
                table = 'hourly'

            else:
                # Use raw data (15-minute resolution)
                query = """
                    SELECT
                        time,
                        energy_kwh,
                        power_kw,
                        voltage_v
                    FROM meter_readings
                    WHERE meter_id = %s
                      AND time BETWEEN %s AND %s
                    ORDER BY time
                """
                table = 'raw'

            logger.info(f"Query {table} table for {meter_id}: " +
                       f"{start_time} to {end_time}")

            cursor = self.db.cursor()
            cursor.execute(query, [meter_id, start_time, end_time])

            return cursor.fetchall()

        def _to_csv(self, readings):
            """Convert readings to CSV format for COPY"""
            import io

            csv_buffer = io.StringIO()
            for r in readings:
                csv_buffer.write(
                    f"{r['meter_id']},{r['time']}," +
                    f"{r['energy_kwh']},{r['power_kw']}," +
                    f"{r['voltage_v']},{r['current_a']}," +
                    f"{r['power_factor']},{r['status']}\n"
                )

            csv_buffer.seek(0)
            return csv_buffer
    ```

    **Compression Results:**

    ```
    Uncompressed data (31 bytes per reading):
    - 960M readings/day × 31 bytes = 29.8 GB/day

    With TimescaleDB compression (columnar + delta encoding):
    - Compression ratio: 8:1
    - Compressed: 29.8 GB / 8 = 3.7 GB/day
    - Savings: 26.1 GB/day (87% reduction)

    Why compression works well:
    1. Meter IDs: Repeated values (dictionary encoding)
    2. Timestamps: Regular intervals (delta-of-delta encoding)
    3. Energy values: Slowly changing (delta encoding)
    4. Voltage/current: Limited range (bit packing)
    ```

    ---

    ### Data Aggregation Strategy

    ```python
    class UsageAggregator:
        """
        Aggregate meter readings at multiple time granularities

        Aggregation levels:
        - 15 minutes: Raw data (real-time monitoring)
        - 1 hour: Continuous aggregate (dashboard charts)
        - 1 day: Continuous aggregate (monthly reports)
        - 1 month: On-demand aggregate (yearly trends)
        """

        def calculate_daily_summary(self, meter_id, date):
            """
            Calculate daily summary with peak/off-peak breakdown

            Time-of-Use (TOU) periods:
            - Peak hours: 14:00-20:00 weekdays (high rate)
            - Off-peak hours: All other times (low rate)
            """
            query = """
                SELECT
                    -- Total consumption
                    SUM(energy_kwh) as total_kwh,

                    -- Peak period (14:00-20:00 weekdays)
                    SUM(CASE
                        WHEN EXTRACT(HOUR FROM time) BETWEEN 14 AND 19
                         AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
                        THEN energy_kwh
                        ELSE 0
                    END) as peak_kwh,

                    -- Off-peak period
                    SUM(CASE
                        WHEN NOT (EXTRACT(HOUR FROM time) BETWEEN 14 AND 19
                                  AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5)
                        THEN energy_kwh
                        ELSE 0
                    END) as off_peak_kwh,

                    -- Peak demand (max power)
                    MAX(power_kw) as peak_demand_kw,

                    -- Average power
                    AVG(power_kw) as avg_power_kw,

                    -- Load factor (efficiency metric)
                    AVG(power_kw) / NULLIF(MAX(power_kw), 0) as load_factor,

                    -- Number of readings (data quality)
                    COUNT(*) as reading_count,

                    -- Expected readings (96 per day)
                    96 as expected_count,

                    -- Data completeness
                    (COUNT(*) * 100.0 / 96) as completeness_percent

                FROM meter_readings
                WHERE meter_id = %s
                  AND time >= %s::date
                  AND time < (%s::date + INTERVAL '1 day')
                GROUP BY meter_id
            """

            result = self.db.execute(query, [meter_id, date, date])[0]

            return {
                'date': date,
                'total_kwh': float(result['total_kwh']),
                'peak_kwh': float(result['peak_kwh']),
                'off_peak_kwh': float(result['off_peak_kwh']),
                'peak_demand_kw': float(result['peak_demand_kw']),
                'avg_power_kw': float(result['avg_power_kw']),
                'load_factor': float(result['load_factor']),
                'data_quality': {
                    'completeness_percent': float(result['completeness_percent']),
                    'missing_readings': 96 - result['reading_count']
                }
            }

        def calculate_hourly_cost(self, meter_id, date, rate_plan):
            """
            Calculate hourly cost using TOU rates

            TOU pricing example:
            - Peak rate: $0.28/kWh (14:00-20:00 weekdays)
            - Off-peak rate: $0.12/kWh (all other times)
            """
            query = """
                SELECT
                    time_bucket('1 hour', time) as hour,
                    SUM(energy_kwh) as total_kwh,
                    EXTRACT(HOUR FROM time) as hour_of_day,
                    EXTRACT(DOW FROM time) as day_of_week
                FROM meter_readings
                WHERE meter_id = %s
                  AND time >= %s::date
                  AND time < (%s::date + INTERVAL '1 day')
                GROUP BY time_bucket('1 hour', time), hour_of_day, day_of_week
                ORDER BY hour
            """

            hourly_data = self.db.execute(query, [meter_id, date, date])

            costs = []
            for row in hourly_data:
                # Determine if peak or off-peak
                is_peak = (
                    14 <= row['hour_of_day'] < 20 and
                    1 <= row['day_of_week'] <= 5  # Monday-Friday
                )

                rate = rate_plan['peak_rate'] if is_peak else rate_plan['off_peak_rate']
                cost = row['total_kwh'] * rate

                costs.append({
                    'hour': row['hour'],
                    'kwh': float(row['total_kwh']),
                    'rate': rate,
                    'cost': cost,
                    'period': 'peak' if is_peak else 'off-peak'
                })

            return costs
    ```

    ---

    ## 3.2 Billing Engine

    ```python
    class BillingEngine:
        """
        Calculate electricity bills with multiple rate structures

        Supported rate types:
        1. Flat rate: Fixed $/kWh
        2. Time-of-Use (TOU): Different rates for peak/off-peak
        3. Tiered: Rate increases with consumption levels
        4. Demand charge: Based on peak power demand
        """

        def calculate_bill(self, meter_id, start_date, end_date, rate_plan):
            """
            Calculate bill for billing period

            Bill components:
            1. Energy charges (kWh × rate)
            2. Demand charges (peak kW × rate)
            3. Fixed charges (monthly fee)
            4. Taxes and fees
            """
            # Get consumption data
            consumption = self.get_consumption_summary(
                meter_id, start_date, end_date
            )

            if rate_plan['type'] == 'tou':
                bill = self._calculate_tou_bill(consumption, rate_plan)
            elif rate_plan['type'] == 'tiered':
                bill = self._calculate_tiered_bill(consumption, rate_plan)
            else:
                bill = self._calculate_flat_bill(consumption, rate_plan)

            return bill

        def _calculate_tou_bill(self, consumption, rate_plan):
            """
            Time-of-Use billing

            Example:
            - Peak hours (14:00-20:00 weekdays): $0.28/kWh
            - Off-peak hours: $0.12/kWh
            - Demand charge: $12.50/kW
            - Fixed charge: $15.00/month
            """
            # Energy charges
            peak_energy_charge = (
                consumption['peak_kwh'] *
                rate_plan['rates']['peak']['rate']
            )

            off_peak_energy_charge = (
                consumption['off_peak_kwh'] *
                rate_plan['rates']['off_peak']['rate']
            )

            total_energy_charge = peak_energy_charge + off_peak_energy_charge

            # Demand charge (based on peak kW)
            demand_charge = (
                consumption['peak_demand_kw'] *
                rate_plan['rates']['demand_charge']
            )

            # Fixed charge
            fixed_charge = rate_plan['rates']['fixed_charge']

            # Subtotal
            subtotal = total_energy_charge + demand_charge + fixed_charge

            # Taxes (example: 10%)
            taxes = subtotal * 0.10

            # Total
            total = subtotal + taxes

            return {
                'consumption': {
                    'total_kwh': consumption['total_kwh'],
                    'peak_kwh': consumption['peak_kwh'],
                    'off_peak_kwh': consumption['off_peak_kwh'],
                    'peak_demand_kw': consumption['peak_demand_kw']
                },
                'charges': {
                    'energy_charge': {
                        'peak_hours': {
                            'kwh': consumption['peak_kwh'],
                            'rate': rate_plan['rates']['peak']['rate'],
                            'amount': peak_energy_charge
                        },
                        'off_peak_hours': {
                            'kwh': consumption['off_peak_kwh'],
                            'rate': rate_plan['rates']['off_peak']['rate'],
                            'amount': off_peak_energy_charge
                        }
                    },
                    'demand_charge': {
                        'peak_demand_kw': consumption['peak_demand_kw'],
                        'rate': rate_plan['rates']['demand_charge'],
                        'amount': demand_charge
                    },
                    'fixed_charge': fixed_charge,
                    'subtotal': subtotal,
                    'taxes': taxes,
                    'total': total
                },
                'rate_plan': rate_plan
            }

        def _calculate_tiered_bill(self, consumption, rate_plan):
            """
            Tiered billing (increasing block rates)

            Example tiers:
            - First 500 kWh: $0.10/kWh
            - Next 500 kWh: $0.15/kWh
            - Above 1000 kWh: $0.20/kWh
            """
            total_kwh = consumption['total_kwh']
            tiers = rate_plan['rates']['tiers']

            energy_charge = 0
            remaining_kwh = total_kwh
            tier_breakdown = []

            for tier in tiers:
                if remaining_kwh <= 0:
                    break

                # Calculate kWh in this tier
                tier_kwh = min(remaining_kwh, tier['limit'] - tier['start'])
                tier_cost = tier_kwh * tier['rate']

                energy_charge += tier_cost
                remaining_kwh -= tier_kwh

                tier_breakdown.append({
                    'tier': tier['name'],
                    'kwh': tier_kwh,
                    'rate': tier['rate'],
                    'amount': tier_cost
                })

            # Fixed charge
            fixed_charge = rate_plan['rates']['fixed_charge']

            # Subtotal and taxes
            subtotal = energy_charge + fixed_charge
            taxes = subtotal * 0.10
            total = subtotal + taxes

            return {
                'consumption': {
                    'total_kwh': total_kwh
                },
                'charges': {
                    'energy_charge': {
                        'tiers': tier_breakdown,
                        'total': energy_charge
                    },
                    'fixed_charge': fixed_charge,
                    'subtotal': subtotal,
                    'taxes': taxes,
                    'total': total
                },
                'rate_plan': rate_plan
            }

        def get_consumption_summary(self, meter_id, start_date, end_date):
            """
            Get consumption summary for billing period

            Query optimized using continuous aggregates
            """
            query = """
                SELECT
                    SUM(total_energy_kwh) as total_kwh,
                    SUM(CASE
                        WHEN EXTRACT(HOUR FROM bucket) BETWEEN 14 AND 19
                         AND EXTRACT(DOW FROM bucket) BETWEEN 1 AND 5
                        THEN total_energy_kwh
                        ELSE 0
                    END) as peak_kwh,
                    SUM(CASE
                        WHEN NOT (EXTRACT(HOUR FROM bucket) BETWEEN 14 AND 19
                                  AND EXTRACT(DOW FROM bucket) BETWEEN 1 AND 5)
                        THEN total_energy_kwh
                        ELSE 0
                    END) as off_peak_kwh,
                    MAX(peak_power_kw) as peak_demand_kw
                FROM meter_readings_hourly
                WHERE meter_id = %s
                  AND bucket >= %s
                  AND bucket < %s
            """

            result = self.db.execute(query, [meter_id, start_date, end_date])[0]

            return {
                'total_kwh': float(result['total_kwh'] or 0),
                'peak_kwh': float(result['peak_kwh'] or 0),
                'off_peak_kwh': float(result['off_peak_kwh'] or 0),
                'peak_demand_kw': float(result['peak_demand_kw'] or 0)
            }
    ```

    **Billing Calculation Example:**

    ```
    Billing period: January 1-31, 2025
    Total consumption: 875.5 kWh
    Peak consumption: 285.3 kWh (14:00-20:00 weekdays)
    Off-peak consumption: 590.2 kWh
    Peak demand: 6.8 kW

    Rate plan: Residential Time-of-Use
    - Peak rate: $0.28/kWh
    - Off-peak rate: $0.12/kWh
    - Demand charge: $12.50/kW
    - Fixed charge: $15.00/month

    Calculation:
    1. Peak energy charge: 285.3 kWh × $0.28 = $79.88
    2. Off-peak energy charge: 590.2 kWh × $0.12 = $70.82
    3. Demand charge: 6.8 kW × $12.50 = $85.00
    4. Fixed charge: $15.00
    5. Subtotal: $250.70
    6. Taxes (10%): $25.07
    7. Total: $275.77

    Average rate: $275.77 / 875.5 kWh = $0.315/kWh
    ```

    ---

    ## 3.3 Anomaly Detection

    ```python
    class AnomalyDetector:
        """
        Detect unusual consumption patterns using statistical methods and ML

        Anomaly types:
        1. Spike: Sudden increase in power (appliance failure, hvac issue)
        2. Drop: Sudden decrease (power outage, equipment offline)
        3. Sustained high: Continuously high usage (stuck equipment)
        4. Unusual pattern: Different from historical baseline
        """

        def __init__(self, tsdb, ml_model):
            self.tsdb = tsdb
            self.ml_model = ml_model
            self.baseline_window_days = 30

        def detect_anomalies(self, meter_id, reading):
            """
            Detect anomalies in real-time as readings arrive

            Multi-level detection:
            1. Rule-based: Simple threshold checks (fast, high precision)
            2. Statistical: Compare to historical baseline (medium speed)
            3. ML-based: Complex pattern detection (slower, high recall)
            """
            anomalies = []

            # Level 1: Rule-based detection (instant)
            rule_anomaly = self._detect_rule_based(reading)
            if rule_anomaly:
                anomalies.append(rule_anomaly)

            # Level 2: Statistical detection (< 100ms)
            stat_anomaly = self._detect_statistical(meter_id, reading)
            if stat_anomaly:
                anomalies.append(stat_anomaly)

            # Level 3: ML-based detection (< 500ms)
            ml_anomaly = self._detect_ml_based(meter_id, reading)
            if ml_anomaly:
                anomalies.append(ml_anomaly)

            return anomalies

        def _detect_rule_based(self, reading):
            """
            Simple rule-based anomaly detection

            Rules:
            1. Power > 20 kW (residential max)
            2. Voltage < 110V or > 130V (out of spec)
            3. Power factor < 70% (inefficient)
            4. Zero consumption for > 24 hours (possible outage)
            """
            # Check power spike
            if reading['power_kw'] > 20:
                return {
                    'type': 'spike',
                    'severity': 'high',
                    'reason': 'Power exceeds residential maximum',
                    'expected': 5.0,
                    'actual': reading['power_kw'],
                    'confidence': 1.0  # Rule-based = 100% confidence
                }

            # Check voltage
            if reading['voltage_v'] < 110 or reading['voltage_v'] > 130:
                return {
                    'type': 'voltage_issue',
                    'severity': 'medium',
                    'reason': 'Voltage out of specification',
                    'expected': 120,
                    'actual': reading['voltage_v'],
                    'confidence': 1.0
                }

            # Check power factor
            if reading['power_factor'] < 70:
                return {
                    'type': 'low_power_factor',
                    'severity': 'low',
                    'reason': 'Inefficient power usage',
                    'expected': 90,
                    'actual': reading['power_factor'],
                    'confidence': 1.0
                }

            return None

        def _detect_statistical(self, meter_id, reading):
            """
            Statistical anomaly detection using historical baseline

            Method: Z-score (standard deviations from mean)
            - Normal: |z| < 2 (95% of data)
            - Anomaly: |z| > 3 (99.7% threshold)
            """
            # Get historical baseline for same time of day
            hour_of_day = reading['time'].hour
            day_of_week = reading['time'].weekday()

            # Query historical data (last 30 days, same hour/day)
            query = """
                SELECT
                    AVG(power_kw) as mean_power,
                    STDDEV(power_kw) as stddev_power,
                    COUNT(*) as sample_count
                FROM meter_readings
                WHERE meter_id = %s
                  AND EXTRACT(HOUR FROM time) = %s
                  AND EXTRACT(DOW FROM time) = %s
                  AND time >= NOW() - INTERVAL '30 days'
                  AND time < NOW() - INTERVAL '1 hour'  -- Exclude recent data
            """

            baseline = self.tsdb.execute(
                query,
                [meter_id, hour_of_day, day_of_week]
            )[0]

            if baseline['sample_count'] < 10:
                # Not enough historical data
                return None

            mean = baseline['mean_power']
            stddev = baseline['stddev_power']

            if stddev < 0.1:
                # Very stable usage, use absolute threshold
                stddev = 0.5

            # Calculate z-score
            z_score = (reading['power_kw'] - mean) / stddev

            # Detect anomaly (|z| > 3 = 99.7% threshold)
            if abs(z_score) > 3:
                return {
                    'type': 'spike' if z_score > 0 else 'drop',
                    'severity': 'high' if abs(z_score) > 5 else 'medium',
                    'reason': f'Usage deviates {abs(z_score):.1f} std devs from baseline',
                    'expected': mean,
                    'actual': reading['power_kw'],
                    'confidence': min(abs(z_score) / 5, 0.99),
                    'baseline': {
                        'mean': mean,
                        'stddev': stddev,
                        'sample_count': baseline['sample_count']
                    }
                }

            return None

        def _detect_ml_based(self, meter_id, reading):
            """
            ML-based anomaly detection using trained model

            Features:
            1. Current reading (power, voltage, current)
            2. Recent history (last 24 hours)
            3. Time features (hour, day, month)
            4. Weather features (temperature)
            5. Historical statistics (mean, stddev)

            Model: Isolation Forest or Autoencoder
            """
            # Get recent history
            recent_readings = self.tsdb.execute("""
                SELECT power_kw, voltage_v, current_a
                FROM meter_readings
                WHERE meter_id = %s
                  AND time >= %s - INTERVAL '24 hours'
                  AND time < %s
                ORDER BY time DESC
                LIMIT 96
            """, [meter_id, reading['time'], reading['time']])

            # Get weather data
            weather = self._get_weather_data(reading['time'])

            # Extract features
            features = {
                # Current reading
                'power_kw': reading['power_kw'],
                'voltage_v': reading['voltage_v'],
                'current_a': reading['current_a'],

                # Time features
                'hour': reading['time'].hour,
                'day_of_week': reading['time'].weekday(),
                'month': reading['time'].month,
                'is_weekend': reading['time'].weekday() >= 5,

                # Recent history statistics
                'recent_mean': np.mean([r['power_kw'] for r in recent_readings]),
                'recent_stddev': np.std([r['power_kw'] for r in recent_readings]),
                'recent_max': np.max([r['power_kw'] for r in recent_readings]),

                # Weather
                'temperature_f': weather['temperature_f'],
                'humidity': weather['humidity']
            }

            # Predict using ML model
            feature_vector = self._to_feature_vector(features)
            anomaly_score = self.ml_model.predict_anomaly_score(feature_vector)

            # Threshold: > 0.8 = anomaly
            if anomaly_score > 0.8:
                return {
                    'type': 'pattern_anomaly',
                    'severity': 'medium',
                    'reason': 'ML model detected unusual pattern',
                    'actual': reading['power_kw'],
                    'confidence': anomaly_score,
                    'features': features
                }

            return None

        def _get_weather_data(self, timestamp):
            """Get weather data for context"""
            # In production, integrate with weather API
            # For now, return mock data
            return {
                'temperature_f': 72,
                'humidity': 65
            }
    ```

    **Anomaly Examples:**

    ```
    Example 1: Power spike
    - Expected power: 2.5 kW (baseline)
    - Actual power: 8.2 kW
    - Deviation: 228% (5.7 standard deviations)
    - Likely cause: HVAC stuck on, water heater malfunction
    - Alert: "Unusual spike in power usage"

    Example 2: Sustained high usage
    - Baseline daily: 30 kWh/day
    - Actual: 65 kWh/day for 3 consecutive days
    - Deviation: 117% increase
    - Likely cause: Pool pump running continuously, forgot to turn off heater
    - Alert: "Your usage is 2x higher than usual"

    Example 3: Zero consumption
    - Expected: 0.5-5 kW (always some base load)
    - Actual: 0 kW for 6 hours
    - Likely cause: Power outage, circuit breaker tripped
    - Alert: "Possible power outage detected"
    ```

    ---

    ## 3.4 Data Collection & Network Reliability

    ```python
    class MeterDataCollector:
        """
        Collect meter readings via cellular/WiFi with retry logic

        Network challenges:
        1. Intermittent connectivity (cellular weak spots)
        2. Network congestion (all meters reporting at once)
        3. Meter device resource constraints (limited memory, battery)

        Solutions:
        1. Local buffering on meter device
        2. Exponential backoff retry
        3. Adaptive upload intervals
        4. Data compression
        """

        def __init__(self):
            self.max_buffer_size = 100  # Buffer up to 100 readings
            self.retry_delays = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
            self.max_retries = 4

        def collect_and_upload(self, meter_id, reading):
            """
            Collect reading and upload with retry logic

            Flow:
            1. Take reading every 15 minutes
            2. Buffer locally
            3. Attempt upload (batch of 4 readings = 1 hour)
            4. On failure, retry with exponential backoff
            5. On success, clear buffer
            """
            # Add to local buffer
            buffer = self._get_buffer(meter_id)
            buffer.append(reading)

            # Upload when buffer reaches batch size (4 readings = 1 hour)
            if len(buffer) >= 4:
                success = self._upload_with_retry(meter_id, buffer)

                if success:
                    # Clear buffer
                    self._clear_buffer(meter_id)
                    return True
                else:
                    # Check buffer size
                    if len(buffer) >= self.max_buffer_size:
                        # Buffer full, drop oldest readings
                        logger.warning(f"Buffer full for {meter_id}, dropping old data")
                        buffer = buffer[-self.max_buffer_size:]
                        self._save_buffer(meter_id, buffer)

                    return False

            return True

        def _upload_with_retry(self, meter_id, readings):
            """Upload with exponential backoff retry"""
            for attempt in range(self.max_retries):
                try:
                    # Compress readings
                    compressed = self._compress_readings(readings)

                    # Upload via HTTPS
                    response = requests.post(
                        f"{API_URL}/api/v1/readings",
                        headers={
                            'Authorization': f'Bearer {self._get_meter_token(meter_id)}',
                            'Content-Encoding': 'gzip'
                        },
                        data=compressed,
                        timeout=30
                    )

                    if response.status_code == 200:
                        logger.info(f"Uploaded {len(readings)} readings for {meter_id}")
                        return True
                    else:
                        logger.warning(f"Upload failed: {response.status_code}")

                except (requests.Timeout, requests.ConnectionError) as e:
                    logger.warning(f"Network error: {e}")

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt]
                    logger.info(f"Retrying in {delay}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(delay)

            logger.error(f"Upload failed after {self.max_retries} attempts")
            return False

        def _compress_readings(self, readings):
            """Compress readings using gzip (2:1 ratio)"""
            import gzip
            import json

            json_data = json.dumps(readings).encode('utf-8')
            compressed = gzip.compress(json_data)

            compression_ratio = len(json_data) / len(compressed)
            logger.debug(f"Compressed {len(json_data)} bytes to " +
                        f"{len(compressed)} bytes ({compression_ratio:.1f}x)")

            return compressed
    ```

    **Network Reliability Strategy:**

    ```
    Scenario 1: Normal operation
    - Reading interval: 15 minutes
    - Upload batch: 4 readings (1 hour of data)
    - Network success rate: 95%
    - Expected behavior: Upload succeeds immediately

    Scenario 2: Temporary network outage (30 minutes)
    - Meter buffers 2 readings locally
    - Retry after 1 minute: Network still down
    - Retry after 5 minutes: Network restored
    - Upload succeeds, buffer cleared
    - Data loss: 0%

    Scenario 3: Extended network outage (6 hours)
    - Meter buffers 24 readings locally
    - Multiple retry attempts fail
    - Buffer size: 24 readings (< 100 max)
    - Network restored after 6 hours
    - Upload all buffered data
    - Data loss: 0%

    Scenario 4: Very long outage (> 25 hours)
    - Meter buffers 100 readings (max capacity)
    - Network still down
    - New readings start overwriting oldest
    - Data loss: Readings beyond 100 buffer size
    - Data loss rate: 4% (4 readings/hour beyond 100)

    Design trade-off:
    - Larger buffer = More resilience but more memory
    - Our choice: 100 readings buffer (~25 hours)
    - Acceptable for 99.9% of outages
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Data Collection Layer (Stateless):
    - Scale collectors based on meter count
    - Each collector: 50K-100K meters
    - 10M meters = 100-200 collector instances
    - Auto-scale based on queue depth

    Time-Series Database (Stateful):
    - Shard by meter_id hash
    - Each shard: 1-2 TB data
    - 4 TB total = 2-4 shards
    - Replication factor: 3x for availability
    - Scale by adding shards (re-shard gradually)

    Query/API Layer (Stateless):
    - Scale based on query load
    - Each API node: 100-200 qps
    - 700 qps peak = 4-7 API nodes
    - Auto-scale based on p99 latency

    Analytics/Anomaly Detection (Stateless):
    - Partition meters across detection nodes
    - Each node: 1-2K meters
    - 10M meters = 5K-10K nodes (or use stream processing)
    - Alternative: Use managed stream processing (Flink, Kinesis Analytics)
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Batch writes (COPY)** | 100x faster writes (10K → 1M/sec) | Slight latency increase (10s batching) |
    | **Continuous aggregations** | 100x faster queries for hourly/daily data | Storage overhead (20% more) |
    | **TimescaleDB compression** | 8x storage reduction | CPU overhead (5-10%) |
    | **Multi-tier storage** | 95% cost savings | Query complexity increases |
    | **Query cache (Redis)** | 10x faster repeated queries | Stale data (30s TTL) |
    | **Buffering on meter** | 99.9% data reliability | Delayed data (up to 25 hours) |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (10M meters, 15-min intervals):

    Compute:
    - 12 collection nodes × $50 = $600
    - 6 TSDB nodes × $300 = $1,800 (with replication)
    - 7 API nodes × $100 = $700
    - 6 analytics nodes × $150 = $900
    - Total compute: $4,000/month

    Storage:
    - Hot tier (335 GB SSD): $34
    - Warm tier (340 GB SSD): $34
    - Cold tier (42 GB S3): $1
    - Total storage: $69/month (with compression + retention)

    Network:
    - Ingress: 10 Mbps × 330 TB/month = Negligible (< $100)
    - Egress: 1.2 Gbps × 3,900 TB/month = $31,200 (mobile apps)

    External APIs:
    - Weather API: $500/month
    - Payment gateway: $200/month

    Total: $36,569/month ≈ $439K/year

    Per-meter cost: $439K / 10M = $0.044/meter/year

    Optimizations:
    1. CDN for mobile app assets: -$15K
    2. Reserved instances (30% discount): -$1.2K
    3. Aggressive compression (reduce warm storage): -$10
    4. Use spot instances for analytics: -$300

    Optimized Total: $20K/month ≈ $240K/year
    Per-meter cost: $0.024/meter/year
    ```

    ---

    ## Reliability & Fault Tolerance

    ```python
    # System reliability metrics

    # Data collection reliability
    meter_reading_success_rate = 99.9%      # 95% network + retry logic
    data_loss_rate = 0.1%                   # Buffer overflow in extreme cases

    # Database availability
    tsdb_availability = 99.95%              # Replication + automatic failover
    query_availability = 99.99%             # Stateless, multi-instance

    # End-to-end availability
    system_availability = 99.9%             # Min of all components

    # Expected downtime
    yearly_downtime = 365 × 24 × 60 × (1 - 0.999) = 525 minutes = 8.76 hours/year
    ```

    **Failure Scenarios:**

    ```
    Scenario 1: Single collector node failure
    - Impact: 50K-100K meters can't upload
    - Mitigation: Meters buffer locally, retry to other collectors
    - Recovery time: Automatic (load balancer detects, redirects)
    - Data loss: 0%

    Scenario 2: TSDB node failure (with replication)
    - Impact: Read queries slightly slower (fall back to replicas)
    - Mitigation: Automatic failover to replica
    - Recovery time: < 30 seconds
    - Data loss: 0% (replicated)

    Scenario 3: Entire data center failure
    - Impact: All services down in that region
    - Mitigation: Multi-region deployment
    - Recovery time: DNS failover to backup region (5-10 minutes)
    - Data loss: < 5 minutes of data (async replication lag)

    Scenario 4: Kafka message queue failure
    - Impact: New readings can't be processed
    - Mitigation: Meters buffer, retry after recovery
    - Recovery time: Restart Kafka cluster (10-30 minutes)
    - Data loss: 0% (meters buffer up to 25 hours)
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you handle network failures for meter data collection?"**
   - Local buffering on meter device (100 readings = 25 hours)
   - Exponential backoff retry (1m, 5m, 15m, 1h)
   - Batch uploads to reduce network overhead
   - Message queue (Kafka) for decoupling and replay
   - **Result:** 99.9% data reliability even with intermittent connectivity

2. **"How do you optimize time-series data storage?"**
   - TimescaleDB with automatic time-based partitioning
   - Columnar compression (8:1 ratio)
   - Continuous aggregations (pre-compute hourly/daily)
   - Multi-tier storage (hot/warm/cold with downsampling)
   - Automatic retention policies (drop old data)
   - **Result:** 95% storage cost savings

3. **"How do you calculate bills with time-of-use rates?"**
   - Query hourly aggregations from continuous aggregate views
   - Classify each hour as peak or off-peak based on time/day
   - Apply appropriate rate to each period
   - Add demand charges (based on peak kW)
   - Include fixed charges and taxes
   - Cache calculations (bills don't change after generation)

4. **"How do you detect anomalies in consumption?"**
   - Multi-level approach: Rule-based, Statistical, ML-based
   - Rule-based: Simple thresholds (power > 20 kW)
   - Statistical: Z-score vs. historical baseline (> 3 std deviations)
   - ML-based: Isolation Forest on features (time, weather, history)
   - Context-aware: Compare to same hour/day in past 30 days
   - **Result:** 95% true positive rate, < 5% false positives

5. **"How do you scale to 10M+ meters?"**
   - Shard time-series database by meter_id hash
   - Horizontal scaling of stateless services (collectors, API, analytics)
   - Use continuous aggregations to reduce query load
   - Implement query caching (Redis)
   - Multi-region deployment for geographic distribution
   - Stream processing for real-time analytics (Kafka Streams, Flink)

6. **"What if all meters report at the same time (thundering herd)?"**
   - Stagger reporting times (random offset per meter)
   - Use message queue (Kafka) to absorb bursts
   - Batch writer buffers and batches inserts
   - Auto-scale collectors based on queue depth
   - Rate limiting per collector node
   - **Result:** Smooth load distribution

7. **"How do you ensure billing accuracy?"**
   - Strong consistency for billing calculations (PostgreSQL ACID)
   - Validate all readings (sanity checks, deduplication)
   - Store complete audit trail (all raw readings)
   - Reconciliation jobs (compare calculated vs. actual)
   - Manual review for high-value anomalies
   - Idempotent billing (same input = same output)

**Key Points to Mention:**

- Time-series database optimized for meter readings (TimescaleDB)
- Multi-tier storage with compression and retention (95% cost savings)
- Continuous aggregations for fast queries (100x faster)
- Local buffering on meters for network reliability (99.9% data)
- Multi-level anomaly detection (rule-based, statistical, ML)
- Time-of-use billing with peak/off-peak rates
- Horizontal scaling for all stateless services
- Event-driven architecture with Kafka

---

## Real-World Examples

**Sense Energy Monitor:**
- Real-time electricity monitoring
- Machine learning for appliance detection
- Mobile app with push notifications
- Detects always-on devices ("vampire loads")
- Solar production tracking
- Architecture: Edge device → Cloud API → Mobile app

**Itron Smart Meters:**
- Deployed by major utilities (PG&E, National Grid)
- Cellular connectivity (LTE-M, NB-IoT)
- AMI (Advanced Metering Infrastructure)
- Mesh networking for reliability
- 15-minute interval data
- Remote disconnect capability

**Emporia Vue:**
- DIY smart meter for existing homes
- CT clamps for circuit-level monitoring
- WiFi connectivity
- Real-time dashboard
- Budget alerts and goals
- Integration with solar inverters

**Pacific Gas & Electric (PG&E):**
- 5.4 million smart meters in California
- 15-minute interval data
- Time-of-use rate plans
- Green Button data export (industry standard)
- Demand response programs
- Outage detection and restoration

---

## Summary

**System Characteristics:**

- **Scale:** 10M meters, 960M readings/day, 11K readings/sec
- **Latency:** < 5 minutes ingestion, < 2s queries
- **Storage:** 4 TB total (with compression + retention)
- **Availability:** 99.95% uptime
- **Retention:** 3 years (multi-tier with downsampling)

**Core Components:**

1. **Data Collectors:** Cellular/WiFi gateways with retry logic
2. **Message Queue (Kafka):** Buffering and replay capability
3. **Time-Series Database (TimescaleDB):** Compressed storage with continuous aggregations
4. **Billing Engine:** TOU rate calculation with multiple rate structures
5. **Anomaly Detector:** Multi-level detection (rule-based, statistical, ML)
6. **Query Engine:** Fast aggregation queries with caching
7. **Mobile App:** Real-time monitoring and alerts

**Key Design Decisions:**

- TimescaleDB for time-series (PostgreSQL extension, mature, SQL queries)
- Multi-tier storage with compression (95% cost savings)
- Continuous aggregations (100x faster queries)
- Local buffering on meters (99.9% data reliability)
- Batch writes with COPY (100x faster than individual INSERTs)
- Multi-level anomaly detection (high accuracy, low false positives)
- Time-of-use billing (reflect true cost of electricity)
- Event-driven architecture (Kafka for decoupling)
- Eventual consistency for readings (strong consistency for billing)

This design provides a scalable, reliable smart meter system capable of monitoring millions of meters in real-time, calculating accurate bills, and detecting anomalies to help users save energy and money.
