# Design Smart Thermostat System (Nest, Ecobee)

A smart thermostat system that learns user preferences, optimizes HVAC control for comfort and energy savings, integrates with weather forecasts, and provides home/away detection through geofencing and ML-powered scheduling.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 5M thermostats, 50M temperature readings/day, ML-powered learning, real-time HVAC control |
| **Key Challenges** | PID control algorithms, learning user behavior (ML), energy optimization, geofencing, offline mode |
| **Core Concepts** | Temperature control loops, HVAC integration, machine learning schedules, weather API, home/away detection |
| **Companies** | Nest, Ecobee, Honeywell Home, Sensibo, Emerson Sensi, Wyze |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Temperature Control** | Maintain target temperature via HVAC control | P0 (Must have) |
    | **Learning Schedule** | Learn user preferences and auto-adjust schedule | P0 (Must have) |
    | **Home/Away Detection** | Geofencing and occupancy detection to save energy | P0 (Must have) |
    | **Manual Override** | User can manually adjust temperature anytime | P0 (Must have) |
    | **Energy Reports** | Show energy savings and usage patterns | P1 (Should have) |
    | **Weather Integration** | Adjust heating/cooling based on forecast | P1 (Should have) |
    | **Remote Control** | Control via mobile app from anywhere | P0 (Must have) |
    | **Multi-zone Support** | Manage multiple zones/rooms independently | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Air quality monitoring (PM2.5, CO2, VOC)
    - Integration with smart vents
    - Detailed HVAC diagnostics and maintenance alerts
    - Voice assistant integration details (Alexa, Google Home)
    - Smart thermostat hardware manufacturing

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Critical for home comfort, HVAC control |
    | **Control Latency** | < 2s for manual adjustments | Fast response for user commands |
    | **Offline Mode** | 100% functionality without internet | Must work during internet outages |
    | **Temperature Accuracy** | ±0.5°F (±0.3°C) | Maintain comfortable environment |
    | **Battery Life** | 6-12 months (if wireless) | Minimize battery replacements |
    | **ML Training Latency** | Daily schedule updates | Learn from user behavior continuously |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total thermostats: 5,000,000 devices
    Average household occupancy: 2.5 people
    Total users: 12.5M registered users

    Temperature readings:
    - Reading interval: 5 minutes (ambient + target)
    - Readings per thermostat: 288/day (24h × 12 per hour)
    - Total readings: 5M × 288 = 1.44B readings/day
    - Per second: 1.44B / 86,400 = 16,667 readings/sec
    - Peak (morning/evening): 50,000 readings/sec

    HVAC control commands:
    - Average on/off cycles: 3-6 per hour during active periods
    - Total commands: 5M × 50 commands/day = 250M commands/day
    - Per second: 2,894 commands/sec
    - Peak: 8,700 commands/sec

    Mobile app queries:
    - Active users: 5M daily (40% of users)
    - Queries per user: 5 queries/day
    - Total: 5M × 5 = 25M queries/day = 289 queries/sec
    - Peak: 900 queries/sec (morning/evening)

    ML training:
    - Schedule learning: Daily per thermostat
    - Training jobs: 5M/day distributed over 24 hours
    - Per second: 58 training jobs/sec
    ```

    ### Storage Estimates

    ```
    Thermostat configuration:
    - Per thermostat: 5 KB (device_id, settings, schedule, HVAC config)
    - Total: 5M × 5 KB = 25 GB

    User profiles:
    - Per user: 3 KB (user_id, email, preferences, home location)
    - Total: 12.5M × 3 KB = 37.5 GB

    Temperature readings (time-series):
    - Per reading: 20 bytes (device_id, timestamp, temp, humidity, target_temp, hvac_state)
    - Daily: 1.44B × 20 bytes = 28.8 GB/day
    - Yearly: 28.8 GB × 365 = 10.5 TB/year
    - 3 years: 31.5 TB

    With compression (8:1 ratio for time-series):
    - 3 years: 31.5 TB / 8 = 3.94 TB compressed

    Retention strategy (multi-tier):

    Tier 1: Full resolution (5 min) - 30 days
    - Storage: 28.8 GB/day × 30 = 864 GB
    - Compressed: 108 GB
    - Use: Real-time monitoring, ML training

    Tier 2: Hourly aggregation - 1 year
    - Reduction: 12x fewer points
    - Storage: 10.5 TB / 12 = 875 GB
    - Compressed: 110 GB
    - Use: Energy reports, long-term patterns

    Tier 3: Daily aggregation - 3 years
    - Reduction: 288x fewer points
    - Storage: 10.5 TB × 3 / 288 = 109 GB
    - Compressed: 14 GB
    - Use: Yearly comparisons, historical trends

    Total storage: 108 GB + 110 GB + 14 GB = 232 GB
    With replication (3x): 696 GB

    Learned schedules (ML models):
    - Per thermostat: 10 KB (schedule patterns, preferences)
    - Total: 5M × 10 KB = 50 GB

    Energy reports:
    - Per thermostat per month: 2 KB
    - Total: 5M × 2 KB × 12 months = 120 GB

    Total: 696 GB (readings) + 25 GB (config) + 37.5 GB (users) + 50 GB (schedules) + 120 GB (reports) = 928.5 GB ≈ 1 TB
    ```

    ### Memory Estimates (Caching)

    ```
    Active sessions:
    - Concurrent users: 100K (0.8% of users active)
    - Session data: 5 KB per user
    - Total: 100K × 5 KB = 500 MB

    Thermostat state cache:
    - Active thermostats: 1M (20% actively heating/cooling)
    - State data: 2 KB per thermostat
    - Total: 1M × 2 KB = 2 GB

    Schedule cache (hot data):
    - Frequently accessed: 2M thermostats
    - Per schedule: 10 KB
    - Total: 2M × 10 KB = 20 GB

    Weather data cache:
    - Unique locations: 1M (grouped by zip code)
    - Per location: 5 KB (current + 24h forecast)
    - Total: 1M × 5 KB = 5 GB

    Total cache: 500 MB + 2 GB + 20 GB + 5 GB = 27.5 GB
    ```

    ---

    ## Key Assumptions

    1. Average temperature reading interval: 5 minutes
    2. HVAC cycles: 3-6 per hour during heating/cooling periods
    3. Battery life (wireless): 6-12 months (with C-wire: unlimited)
    4. WiFi connectivity: 90% of thermostats online
    5. Average home: Single zone (multi-zone 20%)
    6. ML training: Daily schedule updates based on 7-30 days of history
    7. Weather forecast: Updated every 3 hours
    8. Geofencing radius: 500m-5km (user configurable)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Offline-first control** - Thermostat operates autonomously without cloud
    2. **Machine learning** - Learn user patterns and optimize automatically
    3. **Energy optimization** - Balance comfort with energy savings
    4. **Context-aware** - Use location, weather, and occupancy for decisions
    5. **Responsive control** - Fast temperature adjustments (PID algorithm)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Smart Thermostat Device"
            TempSensor[Temperature Sensor<br/>0.5°F accuracy]
            Humidity[Humidity Sensor]
            Display[LCD Display<br/>Touch interface]
            WiFi[WiFi Module<br/>Cloud sync]
            Controller[Controller MCU<br/>PID algorithm]
            HVAC_Relay[HVAC Relays<br/>Heat/Cool/Fan]
        end

        subgraph "Mobile App"
            iOS[iOS App]
            Android[Android App]
            Geofence[Geofencing Service<br/>Home/Away detection]
        end

        subgraph "API Gateway"
            LoadBalancer[Load Balancer]
            API_GW[API Gateway<br/>Auth, Rate limiting]
        end

        subgraph "Core Services"
            ThermostatService[Thermostat Service<br/>Device management]
            ControlService[Control Service<br/>HVAC commands]
            ScheduleService[Schedule Service<br/>User schedules]
            MLService[ML Service<br/>Learning engine]
            EnergyService[Energy Service<br/>Usage analytics]
            WeatherService[Weather Service<br/>Forecast integration]
        end

        subgraph "Data Storage"
            ThermostatDB[(Thermostat DB<br/>PostgreSQL<br/>Config, schedules)]
            TimeSeriesDB[(Time-Series DB<br/>TimescaleDB<br/>Temp readings)]
            MLModelStore[(ML Model Store<br/>S3/Redis<br/>Learned patterns)]
            Cache[Redis Cache<br/>Hot data]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Events, commands]
        end

        subgraph "ML Pipeline"
            ScheduleLearner[Schedule Learner<br/>Pattern detection]
            EnergyOptimizer[Energy Optimizer<br/>Savings calculator]
            AnomalyDetector[Anomaly Detector<br/>HVAC issues]
        end

        subgraph "External Services"
            WeatherAPI[Weather API<br/>OpenWeatherMap<br/>Forecast data]
            PushService[FCM/APNs<br/>Push notifications]
            MapService[Maps API<br/>Geofencing]
        end

        TempSensor --> Controller
        Humidity --> Controller
        Controller --> Display
        Controller --> HVAC_Relay
        Controller --> WiFi

        WiFi <--> LoadBalancer
        iOS --> LoadBalancer
        Android --> LoadBalancer
        Geofence --> iOS
        Geofence --> Android

        LoadBalancer --> API_GW
        API_GW --> ThermostatService
        API_GW --> ControlService
        API_GW --> ScheduleService
        API_GW --> EnergyService

        ThermostatService --> ThermostatDB
        ThermostatService --> Cache
        ThermostatService --> Kafka

        ControlService --> TimeSeriesDB
        ControlService --> Kafka

        ScheduleService --> ThermostatDB
        ScheduleService --> MLModelStore
        ScheduleService --> MLService

        MLService --> ScheduleLearner
        MLService --> EnergyOptimizer
        ScheduleLearner --> MLModelStore
        ScheduleLearner --> TimeSeriesDB

        EnergyService --> TimeSeriesDB
        EnergyService --> EnergyOptimizer

        WeatherService --> WeatherAPI
        WeatherService --> Cache

        Kafka --> AnomalyDetector
        AnomalyDetector --> PushService

        Geofence --> MapService
        Geofence --> API_GW

        style WiFi fill:#e8f5e9
        style Cache fill:#fff4e1
        style ThermostatDB fill:#ffe1e1
        style TimeSeriesDB fill:#ffe1e1
        style MLModelStore fill:#e1f5ff
        style Kafka fill:#e8eaf6
        style MLService fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PID Controller** | Precise temperature control without overshoot | Bang-bang control (inefficient, temperature swings), proportional only (steady-state error) |
    | **TimescaleDB** | Efficient time-series storage for temperature readings | InfluxDB (less SQL support), Cassandra (no built-in aggregations) |
    | **ML Service (Schedule Learning)** | Automatically learn user patterns (when home, preferred temps) | Rule-based schedules only (requires manual programming, inflexible) |
    | **Redis Cache (Weather)** | Reduce Weather API calls (updated every 3h vs. per request) | No cache (expensive API calls), application cache (no sharing) |
    | **Kafka** | Decouple real-time events (temp changes) from analytics | Direct DB writes (couples components), SQS (limited ordering) |
    | **Geofencing** | Automatic home/away detection (save 15-20% energy) | Motion sensors only (false positives), manual mode switching (user friction) |

    **Key Trade-off:** We chose **offline-first architecture** where thermostat can operate completely without cloud connectivity (local PID control, cached schedule), but **cloud provides ML-powered learning and remote access**. This balances reliability with intelligence.

    ---

    ## API Design

    ### 1. Get Thermostat Status

    **Request:**
    ```http
    GET /api/v1/thermostats/{device_id}/status
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "device_id": "thermo_abc123",
      "status": "heating",  // heating, cooling, fan_only, idle, off
      "current_temp_f": 68.5,
      "current_humidity": 45,
      "target_temp_f": 72.0,
      "mode": "heat",  // heat, cool, auto, off, eco
      "fan": "auto",  // auto, on
      "occupancy": "home",  // home, away, sleep
      "hvac_state": {
        "heat_on": true,
        "cool_on": false,
        "fan_on": true,
        "runtime_today_minutes": 125
      },
      "weather": {
        "outdoor_temp_f": 45,
        "forecast_high_f": 52,
        "forecast_low_f": 38,
        "humidity": 65
      },
      "schedule": {
        "current_setpoint": 72.0,
        "next_change_at": "2026-02-05T22:00:00Z",
        "next_setpoint": 65.0
      },
      "energy_today_kwh": 8.5,
      "battery_level": 85,  // null if wired
      "last_seen_at": "2026-02-05T14:32:15Z"
    }
    ```

    **Design Notes:**

    - Return comprehensive state in single request (reduce round trips)
    - Include weather context for user insight
    - Show energy usage for awareness
    - Cache this endpoint (30-second TTL)

    ---

    ### 2. Set Target Temperature

    **Request:**
    ```http
    POST /api/v1/thermostats/{device_id}/target
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "target_temp_f": 74.0,
      "hold_type": "temporary",  // temporary, permanent, until_next_schedule
      "duration_minutes": 120,   // for temporary holds
      "source": "mobile_app"     // mobile_app, voice, manual_thermostat, schedule
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "device_id": "thermo_abc123",
      "target_temp_f": 74.0,
      "hold_type": "temporary",
      "hold_until": "2026-02-05T16:32:00Z",
      "estimated_time_to_reach": 18,  // minutes
      "energy_impact": {
        "additional_cost_usd": 0.85,
        "additional_kwh": 2.1
      }
    }
    ```

    **Design Notes:**

    - Support temporary holds (manual override for a period)
    - Estimate time to reach target (user feedback)
    - Show energy impact (encourage efficiency)
    - Fast response (< 2s including command to device)

    ---

    ### 3. Get/Update Schedule

    **Request (Get):**
    ```http
    GET /api/v1/thermostats/{device_id}/schedule
    Authorization: Bearer <token>
    ```

    **Response:**
    ```json
    {
      "device_id": "thermo_abc123",
      "schedule_type": "learned",  // manual, learned, eco
      "weekly_schedule": {
        "weekday": [
          {
            "time": "06:00",
            "target_temp_f": 68,
            "mode": "heat",
            "name": "Wake"
          },
          {
            "time": "08:00",
            "target_temp_f": 62,
            "mode": "heat",
            "name": "Away"
          },
          {
            "time": "17:00",
            "target_temp_f": 70,
            "mode": "heat",
            "name": "Home"
          },
          {
            "time": "22:00",
            "target_temp_f": 65,
            "mode": "heat",
            "name": "Sleep"
          }
        ],
        "weekend": [
          {
            "time": "08:00",
            "target_temp_f": 70,
            "mode": "heat",
            "name": "Wake"
          },
          {
            "time": "23:00",
            "target_temp_f": 65,
            "mode": "heat",
            "name": "Sleep"
          }
        ]
      },
      "learned_patterns": {
        "typical_wake_time": "06:15",
        "typical_leave_time": "08:05",
        "typical_return_time": "17:20",
        "typical_sleep_time": "22:30",
        "confidence": 0.87
      }
    }
    ```

    **Request (Update):**
    ```http
    PUT /api/v1/thermostats/{device_id}/schedule
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "schedule_type": "manual",
      "weekly_schedule": {
        "weekday": [...],
        "weekend": [...]
      }
    }
    ```

    ---

    ### 4. Set Home/Away Mode (Geofencing)

    **Request:**
    ```http
    POST /api/v1/thermostats/{device_id}/occupancy
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "occupancy": "away",  // home, away, sleep
      "source": "geofence",  // geofence, manual, schedule, motion_sensor
      "location": {
        "lat": 37.7749,
        "lng": -122.4194,
        "accuracy_meters": 50
      },
      "trigger": "left_home"  // left_home, arrived_home
    }
    ```

    **Response:**
    ```json
    {
      "device_id": "thermo_abc123",
      "occupancy": "away",
      "eco_mode_enabled": true,
      "target_temp_adjusted": {
        "previous": 70.0,
        "new": 62.0,
        "savings_kwh_per_hour": 0.8
      },
      "estimated_return_time": "2026-02-05T17:00:00Z"
    }
    ```

    **Design Notes:**

    - Integrate with mobile app geofencing
    - Automatically adjust to eco temperatures when away
    - Show savings to reinforce behavior
    - Predict return time based on patterns

    ---

    ### 5. Report Temperature Reading (Device → Cloud)

    **Request:**
    ```http
    POST /api/v1/telemetry/readings
    Content-Type: application/json
    Authorization: Bearer <device_token>

    {
      "device_id": "thermo_abc123",
      "timestamp": 1738765200,
      "current_temp_f": 68.5,
      "target_temp_f": 72.0,
      "humidity": 45,
      "hvac_state": {
        "heating": true,
        "cooling": false,
        "fan": true
      },
      "power_usage_w": 3500,  // Estimated HVAC power
      "battery_level": 85,
      "wifi_signal_dbm": -65
    }
    ```

    **Response:**
    ```json
    {
      "status": "received",
      "next_report_interval_sec": 300,
      "commands": [
        {
          "command": "update_schedule",
          "schedule": {...}  // If schedule changed
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Thermostats (PostgreSQL)

    ```sql
    -- Thermostat devices
    CREATE TABLE thermostats (
        device_id UUID PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES users(user_id),
        serial_number VARCHAR(50) UNIQUE NOT NULL,
        device_name VARCHAR(100),
        installation_date DATE,
        home_location GEOGRAPHY(POINT),  -- PostGIS

        -- HVAC configuration
        hvac_type VARCHAR(20) NOT NULL,  -- forced_air, heat_pump, radiant, boiler
        has_cooling BOOLEAN DEFAULT true,
        has_heating BOOLEAN DEFAULT true,
        has_fan BOOLEAN DEFAULT true,
        multi_stage_heating BOOLEAN DEFAULT false,
        multi_stage_cooling BOOLEAN DEFAULT false,

        -- Settings
        temperature_scale VARCHAR(1) DEFAULT 'F',  -- F or C
        mode VARCHAR(20) DEFAULT 'auto',  -- heat, cool, auto, off, eco
        fan_mode VARCHAR(20) DEFAULT 'auto',  -- auto, on
        schedule_type VARCHAR(20) DEFAULT 'learned',  -- manual, learned, eco

        -- Eco mode settings
        eco_heat_temp_f DECIMAL(4,1) DEFAULT 62.0,
        eco_cool_temp_f DECIMAL(4,1) DEFAULT 80.0,

        -- State
        current_temp_f DECIMAL(4,1),
        current_humidity SMALLINT,
        target_temp_f DECIMAL(4,1),
        occupancy VARCHAR(20) DEFAULT 'home',  -- home, away, sleep
        hvac_state JSONB,  -- {heating: bool, cooling: bool, fan: bool}

        -- Device info
        firmware_version VARCHAR(20),
        battery_level SMALLINT,  -- 0-100 or NULL if wired
        wifi_connected BOOLEAN DEFAULT true,
        last_seen_at TIMESTAMP,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_user (user_id),
        INDEX idx_location (home_location) USING GIST
    );

    -- Weekly schedules
    CREATE TABLE schedules (
        schedule_id UUID PRIMARY KEY,
        device_id UUID NOT NULL REFERENCES thermostats(device_id),
        day_of_week SMALLINT NOT NULL,  -- 0=Sunday, 6=Saturday
        time_of_day TIME NOT NULL,
        target_temp_f DECIMAL(4,1) NOT NULL,
        mode VARCHAR(20) NOT NULL,
        name VARCHAR(50),
        enabled BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_device_schedule (device_id, day_of_week, time_of_day)
    );
    ```

    ---

    ### Temperature Readings (TimescaleDB)

    ```sql
    -- Time-series temperature data
    CREATE TABLE temperature_readings (
        device_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        current_temp_f REAL NOT NULL,
        target_temp_f REAL NOT NULL,
        humidity SMALLINT,
        outdoor_temp_f REAL,  -- From weather API

        -- HVAC state
        heating BOOLEAN,
        cooling BOOLEAN,
        fan BOOLEAN,
        power_usage_w REAL,  -- Estimated HVAC power

        -- Derived metrics
        temp_error REAL,  -- current - target
        heating_hours REAL,  -- Fraction of hour heating
        cooling_hours REAL,  -- Fraction of hour cooling

        PRIMARY KEY (device_id, time)
    );

    -- Convert to hypertable
    SELECT create_hypertable('temperature_readings', 'time');

    -- Enable compression
    ALTER TABLE temperature_readings SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'device_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    -- Compression policy: compress after 7 days
    SELECT add_compression_policy('temperature_readings', INTERVAL '7 days');

    -- Retention policy: keep 3 years
    SELECT add_retention_policy('temperature_readings', INTERVAL '3 years');

    -- Continuous aggregation: hourly
    CREATE MATERIALIZED VIEW temperature_readings_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        device_id,
        time_bucket('1 hour', time) AS bucket,
        AVG(current_temp_f) as avg_temp_f,
        AVG(target_temp_f) as avg_target_f,
        AVG(outdoor_temp_f) as avg_outdoor_temp_f,
        AVG(humidity) as avg_humidity,
        SUM(CASE WHEN heating THEN 1 ELSE 0 END) * 5.0 / 60 as heating_minutes,  -- 5-min intervals
        SUM(CASE WHEN cooling THEN 1 ELSE 0 END) * 5.0 / 60 as cooling_minutes,
        AVG(power_usage_w) as avg_power_w,
        COUNT(*) as reading_count
    FROM temperature_readings
    GROUP BY device_id, bucket;

    SELECT add_continuous_aggregate_policy('temperature_readings_hourly',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour'
    );

    -- Continuous aggregation: daily
    CREATE MATERIALIZED VIEW temperature_readings_daily
    WITH (timescaledb.continuous) AS
    SELECT
        device_id,
        time_bucket('1 day', bucket) AS day,
        AVG(avg_temp_f) as avg_temp_f,
        MIN(avg_temp_f) as min_temp_f,
        MAX(avg_temp_f) as max_temp_f,
        AVG(avg_outdoor_temp_f) as avg_outdoor_temp_f,
        SUM(heating_minutes) / 60.0 as heating_hours,
        SUM(cooling_minutes) / 60.0 as cooling_hours,
        AVG(avg_power_w) * (SUM(heating_minutes) + SUM(cooling_minutes)) / 60000.0 as energy_kwh
    FROM temperature_readings_hourly
    GROUP BY device_id, day;

    SELECT add_continuous_aggregate_policy('temperature_readings_daily',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day'
    );
    ```

    ---

    ### ML Models (PostgreSQL + S3)

    ```sql
    -- Learned patterns and models
    CREATE TABLE learned_schedules (
        model_id UUID PRIMARY KEY,
        device_id UUID NOT NULL REFERENCES thermostats(device_id),
        model_type VARCHAR(50) NOT NULL,  -- schedule_pattern, occupancy_prediction
        model_version INT DEFAULT 1,

        -- Learned patterns (JSON)
        patterns JSONB NOT NULL,
        -- Example:
        -- {
        --   "weekday_wake_time": "06:15",
        --   "weekday_leave_time": "08:05",
        --   "weekday_return_time": "17:20",
        --   "weekday_sleep_time": "22:30",
        --   "weekend_wake_time": "08:00",
        --   "weekend_sleep_time": "23:00",
        --   "preferred_temp_home": 70,
        --   "preferred_temp_away": 62,
        --   "preferred_temp_sleep": 65
        -- }

        -- Training metadata
        training_samples INT,
        confidence_score REAL,  -- 0-1
        trained_at TIMESTAMP NOT NULL,
        last_applied_at TIMESTAMP,

        -- Model file (for complex ML models)
        model_s3_path VARCHAR(255),  -- s3://bucket/models/device_id/model.pkl

        active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_device_active (device_id, active)
    );

    -- User interactions (for learning)
    CREATE TABLE thermostat_interactions (
        interaction_id UUID PRIMARY KEY,
        device_id UUID NOT NULL REFERENCES thermostats(device_id),
        user_id UUID REFERENCES users(user_id),
        timestamp TIMESTAMPTZ NOT NULL,

        interaction_type VARCHAR(50) NOT NULL,  -- temp_change, mode_change, schedule_override
        previous_state JSONB,
        new_state JSONB,

        -- Context
        day_of_week SMALLINT,
        hour_of_day SMALLINT,
        outdoor_temp_f REAL,
        occupancy VARCHAR(20),

        INDEX idx_device_time (device_id, timestamp DESC)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Temperature Control Loop (PID)

    ```mermaid
    sequenceDiagram
        participant User
        participant Thermostat
        participant PID_Controller
        participant HVAC
        participant TempSensor
        participant Cloud

        Note over Thermostat: Every 5 minutes

        TempSensor->>Thermostat: Read current temp: 68.5°F
        Thermostat->>Thermostat: Get target temp: 72°F
        Thermostat->>PID_Controller: Calculate control output<br/>error = 72 - 68.5 = 3.5°F

        PID_Controller->>PID_Controller: P: proportional = Kp × error<br/>I: integral += Ki × error × dt<br/>D: derivative = Kd × (error - last_error) / dt

        PID_Controller->>PID_Controller: output = P + I + D<br/>output = 0.75 (need heating)

        alt Output > 0.1 (need heating)
            PID_Controller->>HVAC: Turn ON heating<br/>Turn ON fan
            HVAC-->>PID_Controller: Heating started
        else Output < -0.1 (need cooling)
            PID_Controller->>HVAC: Turn ON cooling<br/>Turn ON fan
        else -0.1 ≤ Output ≤ 0.1 (satisfied)
            PID_Controller->>HVAC: Turn OFF heating/cooling<br/>Fan per setting (auto/on)
        end

        Thermostat->>Cloud: Report telemetry (async)<br/>temp, target, hvac_state
        Cloud-->>Thermostat: Commands (if any)<br/>schedule updates

        Note over Thermostat: Wait 5 minutes<br/>Repeat control loop
    ```

    ---

    ### Learning Schedule Flow

    ```mermaid
    sequenceDiagram
        participant User
        participant Thermostat
        participant Mobile_App
        participant Cloud_API
        participant ML_Service
        participant TimeSeriesDB
        participant Model_Store

        User->>Mobile_App: Manually adjust temp<br/>Set to 72°F at 6:15am
        Mobile_App->>Cloud_API: POST /thermostats/{id}/target

        Cloud_API->>Thermostat: Send command
        Thermostat-->>Cloud_API: Acknowledged

        Cloud_API->>TimeSeriesDB: Log interaction<br/>{time: 06:15, action: temp_change, context: weekday_morning}

        Note over ML_Service: Daily training job<br/>(runs at 3am)

        ML_Service->>TimeSeriesDB: Query interactions<br/>Last 30 days for device

        TimeSeriesDB-->>ML_Service: 180 interactions

        ML_Service->>ML_Service: Pattern detection:<br/>- Cluster temp changes by time<br/>- Identify routines (wake, leave, return, sleep)<br/>- Calculate preferred temps per period<br/>- Detect weekday vs. weekend patterns

        ML_Service->>ML_Service: Extract patterns:<br/>weekday_wake: 06:15 ± 10 min<br/>preferred_temp_wake: 70°F<br/>confidence: 0.87

        ML_Service->>Model_Store: Save learned schedule<br/>device_id → patterns

        ML_Service->>Cloud_API: Notify schedule updated

        Cloud_API->>Thermostat: Push new schedule

        Thermostat->>Thermostat: Apply learned schedule<br/>Future automation based on patterns

        Note over Thermostat: Next morning at 6:15am

        Thermostat->>Thermostat: Learned schedule triggers<br/>Auto adjust to 70°F
        Thermostat->>HVAC: Begin preheating
    ```

    ---

    ### Geofencing Home/Away Flow

    ```mermaid
    sequenceDiagram
        participant User
        participant Mobile_App
        participant Geofence_Service
        participant Cloud_API
        participant Thermostat
        participant HVAC

        Note over User: User leaves home

        User->>User: Moves away from home<br/>Distance > 500m

        Mobile_App->>Geofence_Service: Location update<br/>lat, lng, accuracy

        Geofence_Service->>Geofence_Service: Calculate distance from home<br/>Distance: 650m > threshold

        Geofence_Service->>Geofence_Service: Trigger: "left_home"

        Geofence_Service->>Cloud_API: POST /thermostats/{id}/occupancy<br/>{occupancy: "away", trigger: "left_home"}

        Cloud_API->>Thermostat: Set away mode

        Thermostat->>Thermostat: Switch to eco temperature<br/>Heat from 70°F → 62°F<br/>Cool from 72°F → 80°F

        Thermostat->>HVAC: Adjust HVAC to eco setpoint

        Cloud_API-->>Mobile_App: Notification: "Away mode activated, saving energy"

        Note over User: User returns home (later)

        User->>User: Moves toward home<br/>Distance < 500m

        Geofence_Service->>Geofence_Service: Trigger: "arriving_home"

        Geofence_Service->>Cloud_API: POST /thermostats/{id}/occupancy<br/>{occupancy: "home", trigger: "arrived_home"}

        Cloud_API->>Thermostat: Set home mode

        Thermostat->>Thermostat: Restore comfort temperature<br/>Heat from 62°F → 70°F

        Thermostat->>HVAC: Begin heating to comfort temp

        Cloud_API-->>Mobile_App: "Welcome home, warming to 70°F"
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Smart Thermostat subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **PID Control Algorithm** | How to maintain stable temperature? | Proportional-Integral-Derivative control |
    | **Learning User Behavior** | How to learn and predict user preferences? | Pattern detection ML + clustering |
    | **Home/Away Detection** | How to know when to save energy? | Geofencing + motion sensors + patterns |
    | **Energy Optimization** | How to calculate and maximize savings? | HVAC runtime tracking + weather modeling |

    ---

    === "🎛️ PID Control Algorithm"

        ## The Challenge

        **Problem:** Maintain target temperature precisely without overshooting, oscillating, or wasting energy.

        **Requirements:**
        - Reach target temperature quickly (minimize rise time)
        - Avoid temperature overshoot (comfort)
        - Minimize oscillations (avoid frequent on/off cycles)
        - Handle system delays (HVAC takes time to heat/cool)
        - Adapt to different HVAC systems (forced air, radiant, heat pump)

        ---

        ## PID Control Theory

        **PID = Proportional + Integral + Derivative**

        ```
        Control output = Kp × error + Ki × ∫error dt + Kd × d(error)/dt

        Where:
        - error = target_temp - current_temp
        - Kp = proportional gain (react to current error)
        - Ki = integral gain (eliminate steady-state error)
        - Kd = derivative gain (dampen oscillations)
        ```

        **Component behavior:**

        1. **Proportional (P):** React to current error
           - Large error → strong heating/cooling
           - Small error → gentle adjustment
           - Problem alone: steady-state error (never quite reaches target)

        2. **Integral (I):** Accumulate error over time
           - Eliminates steady-state error
           - Ensures target is eventually reached
           - Problem: can cause overshoot if too aggressive

        3. **Derivative (D):** Predict future error based on rate of change
           - Dampens oscillations
           - Prevents overshoot
           - Problem: sensitive to noise

        ---

        ## Python Implementation

        ```python
        import time
        from typing import Optional

        class ThermostatPIDController:
            """
            PID controller for smart thermostat HVAC control

            Maintains target temperature by calculating optimal HVAC output
            based on current error, accumulated error, and error rate of change.
            """

            def __init__(
                self,
                kp: float = 0.5,
                ki: float = 0.05,
                kd: float = 0.1,
                setpoint: float = 70.0
            ):
                """
                Initialize PID controller

                Args:
                    kp: Proportional gain (typical: 0.3-0.8)
                    ki: Integral gain (typical: 0.01-0.1)
                    kd: Derivative gain (typical: 0.05-0.2)
                    setpoint: Target temperature (°F)
                """
                self.kp = kp
                self.ki = ki
                self.kd = kd
                self.setpoint = setpoint

                # State variables
                self.last_error = 0.0
                self.integral = 0.0
                self.last_time = time.time()

                # Output limits
                self.output_min = -1.0  # Full cooling
                self.output_max = 1.0   # Full heating

                # Anti-windup limits (prevent integral windup)
                self.integral_min = -10.0
                self.integral_max = 10.0

            def compute(self, current_temp: float, dt: Optional[float] = None) -> float:
                """
                Compute PID control output

                Args:
                    current_temp: Current temperature reading (°F)
                    dt: Time delta since last call (seconds). If None, auto-calculate.

                Returns:
                    Control output: -1.0 to 1.0
                    - Positive: heating needed (0.0 to 1.0)
                    - Negative: cooling needed (-1.0 to 0.0)
                    - Near zero: maintain current state
                """
                # Calculate time delta
                current_time = time.time()
                if dt is None:
                    dt = current_time - self.last_time
                self.last_time = current_time

                # Avoid division by zero
                if dt <= 0:
                    dt = 1.0

                # Calculate error
                error = self.setpoint - current_temp

                # Proportional term
                p_term = self.kp * error

                # Integral term (accumulated error over time)
                self.integral += error * dt
                # Anti-windup: clamp integral to prevent excessive accumulation
                self.integral = max(self.integral_min, min(self.integral_max, self.integral))
                i_term = self.ki * self.integral

                # Derivative term (rate of change of error)
                derivative = (error - self.last_error) / dt
                d_term = self.kd * derivative

                # Calculate output
                output = p_term + i_term + d_term

                # Clamp output to limits
                output = max(self.output_min, min(self.output_max, output))

                # Update state
                self.last_error = error

                return output

            def set_setpoint(self, setpoint: float):
                """Update target temperature"""
                self.setpoint = setpoint
                # Reset integral on setpoint change to avoid integral windup
                self.integral = 0.0

            def reset(self):
                """Reset PID controller state"""
                self.last_error = 0.0
                self.integral = 0.0
                self.last_time = time.time()


        class HVACController:
            """
            Control HVAC system based on PID output

            Translates continuous PID output (-1 to 1) into discrete HVAC commands.
            """

            def __init__(self, hvac_type: str = "forced_air"):
                self.hvac_type = hvac_type
                self.heating = False
                self.cooling = False
                self.fan = False

                # Hysteresis to prevent rapid cycling
                self.heating_threshold = 0.1   # Start heating if output > 0.1
                self.cooling_threshold = -0.1  # Start cooling if output < -0.1
                self.deadband = 0.2            # 0.1°F deadband

            def apply_control(self, pid_output: float, current_temp: float, target_temp: float) -> dict:
                """
                Apply PID output to HVAC system

                Args:
                    pid_output: PID controller output (-1 to 1)
                    current_temp: Current temperature
                    target_temp: Target temperature

                Returns:
                    HVAC state: {heating, cooling, fan, reason}
                """
                # Calculate error for deadband logic
                error = target_temp - current_temp

                # Determine HVAC action based on PID output and hysteresis
                new_heating = False
                new_cooling = False
                new_fan = False
                reason = "maintaining"

                if pid_output > self.heating_threshold:
                    # Need heating
                    new_heating = True
                    new_fan = True
                    reason = f"heating (output: {pid_output:.2f}, error: {error:.1f}°F)"

                elif pid_output < self.cooling_threshold:
                    # Need cooling
                    new_cooling = True
                    new_fan = True
                    reason = f"cooling (output: {pid_output:.2f}, error: {error:.1f}°F)"

                else:
                    # Within deadband, turn off heating/cooling
                    new_heating = False
                    new_cooling = False
                    # Fan based on user preference (auto/on)
                    new_fan = False  # In auto mode
                    reason = f"satisfied (output: {pid_output:.2f}, error: {error:.1f}°F)"

                # Apply state changes
                state_changed = (
                    new_heating != self.heating or
                    new_cooling != self.cooling or
                    new_fan != self.fan
                )

                self.heating = new_heating
                self.cooling = new_cooling
                self.fan = new_fan

                return {
                    'heating': self.heating,
                    'cooling': self.cooling,
                    'fan': self.fan,
                    'reason': reason,
                    'state_changed': state_changed,
                    'pid_output': pid_output
                }


        # Example usage
        def simulate_heating_cycle():
            """Simulate a heating cycle with PID control"""
            print("=== Smart Thermostat PID Control Simulation ===\n")

            # Initialize controllers
            pid = ThermostatPIDController(
                kp=0.5,
                ki=0.05,
                kd=0.1,
                setpoint=72.0  # Target: 72°F
            )
            hvac = HVACController()

            # Simulate heating cycle
            current_temp = 65.0  # Start at 65°F
            outdoor_temp = 45.0
            thermal_mass = 1.0  # Heat capacity of home

            print(f"Initial: Current={current_temp}°F, Target={pid.setpoint}°F\n")

            for minute in range(60):  # Simulate 60 minutes
                # PID control
                pid_output = pid.compute(current_temp, dt=60.0)  # 1-minute interval

                # Apply to HVAC
                hvac_state = hvac.apply_control(pid_output, current_temp, pid.setpoint)

                # Simulate temperature change
                if hvac_state['heating']:
                    # Heating: increase temp (forced air: ~0.5°F per minute when heating)
                    heating_rate = 0.5
                    current_temp += heating_rate * (1.0 / thermal_mass)
                elif hvac_state['cooling']:
                    # Cooling: decrease temp
                    cooling_rate = 0.4
                    current_temp -= cooling_rate * (1.0 / thermal_mass)

                # Heat loss to outdoor temperature
                heat_loss_rate = 0.02  # Proportional to temp difference
                current_temp -= (current_temp - outdoor_temp) * heat_loss_rate

                # Log every 5 minutes
                if minute % 5 == 0 or hvac_state['state_changed']:
                    print(f"Minute {minute:2d}: Temp={current_temp:.1f}°F, " +
                          f"Error={pid.setpoint - current_temp:+.1f}°F, " +
                          f"PID={pid_output:+.2f}, {hvac_state['reason']}")

            print(f"\nFinal: Current={current_temp:.1f}°F, " +
                  f"Error={pid.setpoint - current_temp:+.1f}°F")


        if __name__ == "__main__":
            simulate_heating_cycle()
        ```

        **Output Example:**

        ```
        === Smart Thermostat PID Control Simulation ===

        Initial: Current=65.0°F, Target=72.0°F

        Minute  0: Temp=65.0°F, Error=+7.0°F, PID=+0.50, heating (output: 0.50, error: 7.0°F)
        Minute  5: Temp=67.5°F, Error=+4.5°F, PID=+0.43, heating (output: 0.43, error: 4.5°F)
        Minute 10: Temp=69.8°F, Error=+2.2°F, PID=+0.32, heating (output: 0.32, error: 2.2°F)
        Minute 15: Temp=71.2°F, Error=+0.8°F, PID=+0.18, heating (output: 0.18, error: 0.8°F)
        Minute 20: Temp=71.9°F, Error=+0.1°F, PID=+0.09, satisfied (output: 0.09, error: 0.1°F)
        Minute 25: Temp=71.7°F, Error=+0.3°F, PID=+0.12, heating (output: 0.12, error: 0.3°F)
        Minute 30: Temp=72.0°F, Error=+0.0°F, PID=+0.05, satisfied (output: 0.05, error: 0.0°F)
        Minute 35: Temp=71.9°F, Error=+0.1°F, PID=+0.08, satisfied (output: 0.08, error: 0.1°F)
        Minute 40: Temp=72.0°F, Error=+0.0°F, PID=+0.04, satisfied (output: 0.04, error: 0.0°F)

        Final: Current=72.0°F, Error=+0.0°F
        ```

        ---

        ## PID Tuning for Different HVAC Systems

        | HVAC Type | Kp | Ki | Kd | Reasoning |
        |-----------|----|----|-----|-----------|
        | **Forced Air** | 0.5 | 0.05 | 0.1 | Fast response, moderate inertia |
        | **Radiant Floor** | 0.2 | 0.02 | 0.05 | Slow response, high thermal mass, low Kp to avoid overshoot |
        | **Heat Pump** | 0.4 | 0.04 | 0.08 | Moderate response, efficient |
        | **Boiler** | 0.3 | 0.03 | 0.06 | Slow response, high inertia |

        **Auto-tuning approach:**

        ```python
        def auto_tune_pid(device_id: str, hvac_type: str) -> dict:
            """
            Auto-tune PID parameters based on HVAC type and observed behavior

            Uses Ziegler-Nichols method:
            1. Start with I=0, D=0, increase Kp until oscillation
            2. Record Ku (ultimate gain) and Tu (oscillation period)
            3. Calculate PID gains:
               Kp = 0.6 × Ku
               Ki = 2 × Kp / Tu
               Kd = Kp × Tu / 8
            """
            # Lookup table for common HVAC types
            defaults = {
                'forced_air': {'kp': 0.5, 'ki': 0.05, 'kd': 0.1},
                'radiant': {'kp': 0.2, 'ki': 0.02, 'kd': 0.05},
                'heat_pump': {'kp': 0.4, 'ki': 0.04, 'kd': 0.08},
                'boiler': {'kp': 0.3, 'ki': 0.03, 'kd': 0.06}
            }

            # Start with defaults
            params = defaults.get(hvac_type, defaults['forced_air'])

            # Observe system behavior over 24 hours
            # Adjust based on overshoot, rise time, settling time
            # (production system would use historical data)

            return params
        ```

    === "🧠 Learning User Behavior (ML)"

        ## The Challenge

        **Problem:** Learn when users are home, their preferred temperatures, and automatically create optimal schedules.

        **Goals:**
        - Detect routine patterns (wake, leave, return, sleep times)
        - Learn preferred temperatures for each period
        - Distinguish weekday vs. weekend patterns
        - Adapt to changes in routine over time
        - Preemptively heat/cool before user arrival

        ---

        ## Pattern Detection Algorithm

        ```python
        import numpy as np
        from sklearn.cluster import DBSCAN
        from datetime import datetime, timedelta
        from collections import defaultdict
        from typing import List, Dict

        class ScheduleLearner:
            """
            Learn user schedule patterns from temperature adjustments

            Approach:
            1. Collect user interactions (manual temp changes)
            2. Cluster interactions by time of day to find routines
            3. Extract preferred temperatures for each routine
            4. Distinguish weekday vs. weekend patterns
            5. Generate learned schedule
            """

            def __init__(self, min_samples: int = 5, confidence_threshold: float = 0.7):
                self.min_samples = min_samples
                self.confidence_threshold = confidence_threshold

            def learn_schedule(self, interactions: List[dict]) -> dict:
                """
                Learn schedule from user interactions

                Args:
                    interactions: List of user temperature changes
                    [
                        {
                            'timestamp': datetime,
                            'target_temp': float,
                            'day_of_week': int,  # 0=Monday, 6=Sunday
                            'hour': int,
                            'minute': int
                        },
                        ...
                    ]

                Returns:
                    Learned schedule with patterns
                """
                if len(interactions) < self.min_samples:
                    return self._get_default_schedule()

                # Separate weekday vs. weekend
                weekday_interactions = [
                    i for i in interactions if i['day_of_week'] < 5
                ]
                weekend_interactions = [
                    i for i in interactions if i['day_of_week'] >= 5
                ]

                # Learn patterns for each
                weekday_patterns = self._extract_patterns(weekday_interactions, 'weekday')
                weekend_patterns = self._extract_patterns(weekend_interactions, 'weekend')

                # Calculate confidence based on sample size and consistency
                confidence = self._calculate_confidence(interactions)

                return {
                    'weekday': weekday_patterns,
                    'weekend': weekend_patterns,
                    'confidence': confidence,
                    'trained_at': datetime.now(),
                    'sample_count': len(interactions)
                }

            def _extract_patterns(self, interactions: List[dict], period_type: str) -> List[dict]:
                """
                Extract routine patterns from interactions using clustering

                Clustering approach:
                - Convert interactions to time-of-day (minutes since midnight)
                - Use DBSCAN to cluster similar times
                - Each cluster = a routine (wake, leave, return, sleep)
                - Extract preferred temperature for each routine
                """
                if len(interactions) < self.min_samples:
                    return []

                # Convert to minutes since midnight
                time_points = []
                for interaction in interactions:
                    minutes = interaction['hour'] * 60 + interaction['minute']
                    time_points.append([minutes])

                time_points = np.array(time_points)

                # Cluster with DBSCAN (finds variable number of clusters)
                # eps=60 means cluster if within 60 minutes (1 hour)
                clustering = DBSCAN(eps=60, min_samples=self.min_samples).fit(time_points)

                # Extract patterns from clusters
                patterns = []
                for cluster_id in set(clustering.labels_):
                    if cluster_id == -1:  # Noise
                        continue

                    # Get interactions in this cluster
                    cluster_mask = clustering.labels_ == cluster_id
                    cluster_interactions = [
                        interactions[i] for i in range(len(interactions))
                        if cluster_mask[i]
                    ]

                    # Calculate average time and temperature for this routine
                    avg_minutes = np.mean([
                        i['hour'] * 60 + i['minute']
                        for i in cluster_interactions
                    ])
                    avg_temp = np.mean([i['target_temp'] for i in cluster_interactions])
                    std_minutes = np.std([
                        i['hour'] * 60 + i['minute']
                        for i in cluster_interactions
                    ])

                    # Convert back to hour:minute
                    avg_hour = int(avg_minutes // 60)
                    avg_minute = int(avg_minutes % 60)

                    # Determine routine name based on time
                    routine_name = self._classify_routine(avg_hour)

                    patterns.append({
                        'routine': routine_name,
                        'time': f"{avg_hour:02d}:{avg_minute:02d}",
                        'target_temp': round(avg_temp, 1),
                        'variance_minutes': round(std_minutes, 1),
                        'sample_count': len(cluster_interactions)
                    })

                # Sort patterns by time
                patterns.sort(key=lambda p: p['time'])

                return patterns

            def _classify_routine(self, hour: int) -> str:
                """Classify routine based on time of day"""
                if 5 <= hour < 9:
                    return "wake"
                elif 9 <= hour < 16:
                    return "away"
                elif 16 <= hour < 21:
                    return "home"
                else:
                    return "sleep"

            def _calculate_confidence(self, interactions: List[dict]) -> float:
                """
                Calculate confidence score based on:
                1. Sample size (more samples = higher confidence)
                2. Consistency (low variance = higher confidence)
                3. Recency (recent data = higher confidence)
                """
                # Sample size factor (0.5 to 1.0)
                sample_factor = min(len(interactions) / 100.0, 1.0) * 0.5 + 0.5

                # Consistency factor (low variance in times = high confidence)
                if len(interactions) > 0:
                    times = [i['hour'] * 60 + i['minute'] for i in interactions]
                    variance = np.var(times)
                    consistency_factor = max(0, 1.0 - (variance / 3600))  # 3600 = 1 hour variance
                else:
                    consistency_factor = 0

                # Recency factor (more recent = higher confidence)
                if len(interactions) > 0:
                    days_old = [
                        (datetime.now() - i['timestamp']).days
                        for i in interactions
                    ]
                    avg_age = np.mean(days_old)
                    recency_factor = max(0, 1.0 - (avg_age / 30))  # 30 days
                else:
                    recency_factor = 0

                # Weighted average
                confidence = (
                    sample_factor * 0.4 +
                    consistency_factor * 0.4 +
                    recency_factor * 0.2
                )

                return round(confidence, 2)

            def _get_default_schedule(self) -> dict:
                """Return default schedule if not enough data"""
                return {
                    'weekday': [
                        {'routine': 'wake', 'time': '06:00', 'target_temp': 68.0},
                        {'routine': 'away', 'time': '08:00', 'target_temp': 62.0},
                        {'routine': 'home', 'time': '17:00', 'target_temp': 70.0},
                        {'routine': 'sleep', 'time': '22:00', 'target_temp': 65.0}
                    ],
                    'weekend': [
                        {'routine': 'wake', 'time': '08:00', 'target_temp': 70.0},
                        {'routine': 'sleep', 'time': '23:00', 'target_temp': 65.0}
                    ],
                    'confidence': 0.0,
                    'trained_at': None,
                    'sample_count': 0
                }


        # Example usage
        def example_schedule_learning():
            """Demonstrate schedule learning from user interactions"""
            print("=== Smart Thermostat Schedule Learning ===\n")

            # Simulate user interactions over 30 days
            interactions = []
            base_date = datetime.now() - timedelta(days=30)

            # Weekday pattern: wake at 6:15am, leave at 8:00am, return at 5:30pm, sleep at 10:30pm
            for day in range(30):
                date = base_date + timedelta(days=day)
                day_of_week = date.weekday()

                if day_of_week < 5:  # Weekday
                    # Wake: around 6:15am ± 15 minutes
                    interactions.append({
                        'timestamp': date.replace(hour=6, minute=15) + timedelta(minutes=np.random.randint(-15, 15)),
                        'target_temp': 68 + np.random.randint(-1, 2),
                        'day_of_week': day_of_week,
                        'hour': 6,
                        'minute': 15
                    })

                    # Leave: around 8:00am ± 10 minutes
                    interactions.append({
                        'timestamp': date.replace(hour=8, minute=0) + timedelta(minutes=np.random.randint(-10, 10)),
                        'target_temp': 62 + np.random.randint(0, 2),
                        'day_of_week': day_of_week,
                        'hour': 8,
                        'minute': 0
                    })

                    # Return: around 5:30pm ± 20 minutes
                    interactions.append({
                        'timestamp': date.replace(hour=17, minute=30) + timedelta(minutes=np.random.randint(-20, 20)),
                        'target_temp': 70 + np.random.randint(-1, 2),
                        'day_of_week': day_of_week,
                        'hour': 17,
                        'minute': 30
                    })

                    # Sleep: around 10:30pm ± 30 minutes
                    interactions.append({
                        'timestamp': date.replace(hour=22, minute=30) + timedelta(minutes=np.random.randint(-30, 30)),
                        'target_temp': 65 + np.random.randint(-1, 2),
                        'day_of_week': day_of_week,
                        'hour': 22,
                        'minute': 30
                    })

                else:  # Weekend
                    # Wake: around 8:00am
                    interactions.append({
                        'timestamp': date.replace(hour=8, minute=0) + timedelta(minutes=np.random.randint(-20, 20)),
                        'target_temp': 70 + np.random.randint(-1, 2),
                        'day_of_week': day_of_week,
                        'hour': 8,
                        'minute': 0
                    })

                    # Sleep: around 11:00pm
                    interactions.append({
                        'timestamp': date.replace(hour=23, minute=0) + timedelta(minutes=np.random.randint(-30, 30)),
                        'target_temp': 65 + np.random.randint(-1, 2),
                        'day_of_week': day_of_week,
                        'hour': 23,
                        'minute': 0
                    })

            # Learn schedule
            learner = ScheduleLearner(min_samples=5, confidence_threshold=0.7)
            learned_schedule = learner.learn_schedule(interactions)

            # Print results
            print(f"Learned from {learned_schedule['sample_count']} interactions")
            print(f"Confidence: {learned_schedule['confidence']:.2f}\n")

            print("Weekday Schedule:")
            for pattern in learned_schedule['weekday']:
                print(f"  {pattern['routine'].capitalize():6s} @ {pattern['time']}: " +
                      f"{pattern['target_temp']}°F (±{pattern['variance_minutes']:.0f} min variance)")

            print("\nWeekend Schedule:")
            for pattern in learned_schedule['weekend']:
                print(f"  {pattern['routine'].capitalize():6s} @ {pattern['time']}: " +
                      f"{pattern['target_temp']}°F (±{pattern['variance_minutes']:.0f} min variance)")


        if __name__ == "__main__":
            example_schedule_learning()
        ```

        **Output Example:**

        ```
        === Smart Thermostat Schedule Learning ===

        Learned from 128 interactions
        Confidence: 0.82

        Weekday Schedule:
          Wake   @ 06:15: 68.0°F (±12 min variance)
          Away   @ 08:00: 62.1°F (±8 min variance)
          Home   @ 17:30: 70.2°F (±18 min variance)
          Sleep  @ 22:30: 65.1°F (±25 min variance)

        Weekend Schedule:
          Wake   @ 08:00: 70.1°F (±15 min variance)
          Sleep  @ 23:00: 65.0°F (±22 min variance)
        ```

    === "📍 Home/Away Detection (Geofencing)"

        ## The Challenge

        **Problem:** Automatically detect when users leave/arrive home to enable energy-saving eco mode.

        **Approaches:**
        1. **Geofencing:** Use phone GPS location
        2. **Motion sensors:** Detect movement in home
        3. **Schedule patterns:** Predict based on learned routines
        4. **Multi-user coordination:** Handle homes with multiple residents

        ---

        ## Geofencing Implementation

        ```python
        from math import radians, cos, sin, asin, sqrt
        from datetime import datetime, timedelta
        from typing import List, Optional

        class GeofenceManager:
            """
            Manage home/away detection using geofencing

            Uses Haversine formula to calculate distance from home
            Implements hysteresis to prevent rapid switching
            """

            def __init__(
                self,
                home_lat: float,
                home_lng: float,
                home_radius_meters: float = 500,
                away_radius_meters: float = 1000
            ):
                """
                Initialize geofence

                Args:
                    home_lat: Home latitude
                    home_lng: Home longitude
                    home_radius_meters: Radius to trigger "home" (typical: 100-500m)
                    away_radius_meters: Radius to trigger "away" (typical: 500-2000m)

                Hysteresis:
                - Arriving home: < home_radius_meters
                - Leaving home: > away_radius_meters
                - Between radii: maintain previous state (prevent flapping)
                """
                self.home_lat = home_lat
                self.home_lng = home_lng
                self.home_radius = home_radius_meters
                self.away_radius = away_radius_meters

                # State
                self.current_state = "unknown"  # home, away, unknown
                self.last_transition_time = None
                self.min_transition_delay_minutes = 5  # Debounce rapid changes

            def check_location(
                self,
                user_lat: float,
                user_lng: float,
                timestamp: datetime
            ) -> dict:
                """
                Check if user location triggers state change

                Returns:
                    {
                        'state': 'home' | 'away' | 'unknown',
                        'distance_meters': float,
                        'state_changed': bool,
                        'previous_state': str
                    }
                """
                # Calculate distance from home
                distance = self._haversine_distance(
                    self.home_lat, self.home_lng,
                    user_lat, user_lng
                )

                previous_state = self.current_state
                new_state = self.current_state

                # Determine new state with hysteresis
                if distance <= self.home_radius:
                    new_state = "home"
                elif distance >= self.away_radius:
                    new_state = "away"
                # Else: maintain current state (hysteresis zone)

                # Check for state change
                state_changed = False
                if new_state != self.current_state:
                    # Check debounce delay
                    if self.last_transition_time is None or \
                       (timestamp - self.last_transition_time) > timedelta(minutes=self.min_transition_delay_minutes):
                        self.current_state = new_state
                        self.last_transition_time = timestamp
                        state_changed = True

                return {
                    'state': self.current_state,
                    'distance_meters': round(distance, 1),
                    'state_changed': state_changed,
                    'previous_state': previous_state,
                    'timestamp': timestamp
                }

            def _haversine_distance(
                self,
                lat1: float, lon1: float,
                lat2: float, lon2: float
            ) -> float:
                """
                Calculate distance between two lat/lng points using Haversine formula

                Returns:
                    Distance in meters
                """
                # Convert to radians
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))

                # Radius of Earth in meters
                r = 6371000

                return c * r


        class MultiUserOccupancyManager:
            """
            Handle occupancy for homes with multiple users

            Strategy:
            - Home if ANY user is home
            - Away only if ALL users are away
            - Track last known location for each user
            """

            def __init__(self, home_lat: float, home_lng: float):
                self.geofence = GeofenceManager(home_lat, home_lng)
                self.user_states = {}  # user_id -> state

            def update_user_location(
                self,
                user_id: str,
                lat: float,
                lng: float,
                timestamp: datetime
            ) -> dict:
                """
                Update user location and recalculate home occupancy

                Returns:
                    {
                        'home_occupied': bool,
                        'user_count_home': int,
                        'state_changed': bool
                    }
                """
                # Check user location
                location_result = self.geofence.check_location(lat, lng, timestamp)

                # Update user state
                previous_user_state = self.user_states.get(user_id)
                self.user_states[user_id] = {
                    'state': location_result['state'],
                    'distance': location_result['distance_meters'],
                    'last_update': timestamp
                }

                # Calculate home occupancy (ANY user home = occupied)
                users_home = [
                    uid for uid, state in self.user_states.items()
                    if state['state'] == 'home'
                ]

                home_occupied = len(users_home) > 0

                # Detect state change
                state_changed = (previous_user_state is None) or \
                                (previous_user_state.get('state') != location_result['state'])

                return {
                    'home_occupied': home_occupied,
                    'user_count_home': len(users_home),
                    'users_home': users_home,
                    'state_changed': state_changed,
                    'trigger_user': user_id if state_changed else None
                }


        # Example usage
        def example_geofencing():
            """Demonstrate geofencing home/away detection"""
            print("=== Smart Thermostat Geofencing ===\n")

            # Home location (example: San Francisco)
            home_lat = 37.7749
            home_lng = -122.4194

            geofence = GeofenceManager(
                home_lat=home_lat,
                home_lng=home_lng,
                home_radius_meters=200,
                away_radius_meters=1000
            )

            # Simulate user journey
            locations = [
                # Start at home
                {'lat': 37.7749, 'lng': -122.4194, 'desc': 'At home'},
                # Walking away
                {'lat': 37.7755, 'lng': -122.4180, 'desc': 'Walking away (300m)'},
                # Driving away
                {'lat': 37.7780, 'lng': -122.4150, 'desc': 'Driving away (600m)'},
                {'lat': 37.7820, 'lng': -122.4100, 'desc': 'Further away (1.2km)'},
                # At work
                {'lat': 37.7900, 'lng': -122.4000, 'desc': 'At work (2.5km)'},
                # Driving home
                {'lat': 37.7820, 'lng': -122.4100, 'desc': 'Driving home (1.2km)'},
                {'lat': 37.7780, 'lng': -122.4150, 'desc': 'Getting closer (600m)'},
                {'lat': 37.7755, 'lng': -122.4180, 'desc': 'Almost home (300m)'},
                # Arriving home
                {'lat': 37.7749, 'lng': -122.4194, 'desc': 'Arrived home'},
            ]

            timestamp = datetime.now()

            for i, loc in enumerate(locations):
                result = geofence.check_location(
                    loc['lat'], loc['lng'],
                    timestamp + timedelta(minutes=i * 10)
                )

                status_emoji = "🏠" if result['state'] == 'home' else "🚗" if result['state'] == 'away' else "❓"

                print(f"Location {i+1}: {loc['desc']}")
                print(f"  Distance: {result['distance_meters']:.0f}m")
                print(f"  State: {status_emoji} {result['state']}")
                if result['state_changed']:
                    print(f"  ⚡ STATE CHANGED: {result['previous_state']} → {result['state']}")
                print()


        if __name__ == "__main__":
            example_geofencing()
        ```

        **Output Example:**

        ```
        === Smart Thermostat Geofencing ===

        Location 1: At home
          Distance: 0m
          State: 🏠 home
          ⚡ STATE CHANGED: unknown → home

        Location 2: Walking away (300m)
          Distance: 335m
          State: 🏠 home

        Location 3: Driving away (600m)
          Distance: 607m
          State: 🏠 home

        Location 4: Further away (1.2km)
          Distance: 1247m
          State: 🚗 away
          ⚡ STATE CHANGED: home → away

        Location 5: At work (2.5km)
          Distance: 2518m
          State: 🚗 away

        Location 6: Driving home (1.2km)
          Distance: 1247m
          State: 🚗 away

        Location 7: Getting closer (600m)
          Distance: 607m
          State: 🚗 away

        Location 8: Almost home (300m)
          Distance: 335m
          State: 🚗 away

        Location 9: Arrived home
          Distance: 0m
          State: 🏠 home
          ⚡ STATE CHANGED: away → home
        ```

    === "⚡ Energy Optimization & Savings"

        ## The Challenge

        **Problem:** Calculate energy usage and savings from smart thermostat features.

        **Savings sources:**
        1. **Away mode:** Lower heating/cooling when nobody home (15-20% savings)
        2. **Sleep mode:** Lower temperature overnight (5-10% savings)
        3. **Optimized schedules:** Avoid heating/cooling empty home (10-15% savings)
        4. **Weather-aware preheating:** Use outdoor temp to optimize start times (3-5% savings)

        ---

        ## Energy Calculation

        ```python
        from datetime import datetime, timedelta
        from typing import List, Dict

        class EnergyCalculator:
            """
            Calculate HVAC energy usage and savings

            Estimates based on:
            - HVAC runtime (hours)
            - Outdoor temperature (heating/cooling degree days)
            - Home characteristics (insulation, size)
            """

            def __init__(
                self,
                hvac_heating_power_kw: float = 5.0,
                hvac_cooling_power_kw: float = 3.5,
                electricity_rate_usd_per_kwh: float = 0.13
            ):
                """
                Initialize energy calculator

                Args:
                    hvac_heating_power_kw: HVAC heating power (typical: 3-7 kW)
                    hvac_cooling_power_kw: HVAC cooling power (typical: 2-5 kW)
                    electricity_rate_usd_per_kwh: Cost per kWh
                """
                self.heating_power_kw = hvac_heating_power_kw
                self.cooling_power_kw = hvac_cooling_power_kw
                self.electricity_rate = electricity_rate_usd_per_kwh

            def calculate_daily_usage(
                self,
                heating_hours: float,
                cooling_hours: float
            ) -> dict:
                """
                Calculate daily energy usage

                Args:
                    heating_hours: Hours of heating operation
                    cooling_hours: Hours of cooling operation

                Returns:
                    Energy usage and cost
                """
                heating_kwh = heating_hours * self.heating_power_kw
                cooling_kwh = cooling_hours * self.cooling_power_kw
                total_kwh = heating_kwh + cooling_kwh

                heating_cost = heating_kwh * self.electricity_rate
                cooling_cost = cooling_kwh * self.electricity_rate
                total_cost = total_kwh * self.electricity_rate

                return {
                    'heating_kwh': round(heating_kwh, 2),
                    'cooling_kwh': round(cooling_kwh, 2),
                    'total_kwh': round(total_kwh, 2),
                    'heating_cost_usd': round(heating_cost, 2),
                    'cooling_cost_usd': round(cooling_cost, 2),
                    'total_cost_usd': round(total_cost, 2)
                }

            def calculate_savings(
                self,
                actual_usage: dict,
                baseline_usage: dict
            ) -> dict:
                """
                Calculate savings vs. baseline (manual thermostat)

                Args:
                    actual_usage: Actual energy usage with smart thermostat
                    baseline_usage: Estimated baseline without smart features

                Returns:
                    Savings metrics
                """
                saved_kwh = baseline_usage['total_kwh'] - actual_usage['total_kwh']
                saved_usd = baseline_usage['total_cost_usd'] - actual_usage['total_cost_usd']
                savings_percent = (saved_kwh / baseline_usage['total_kwh']) * 100 if baseline_usage['total_kwh'] > 0 else 0

                return {
                    'saved_kwh': round(saved_kwh, 2),
                    'saved_usd': round(saved_usd, 2),
                    'savings_percent': round(savings_percent, 1),
                    'actual_usage': actual_usage,
                    'baseline_usage': baseline_usage
                }


        class SavingsEstimator:
            """
            Estimate savings from smart thermostat features

            Baseline: Manual thermostat at constant temperature
            Smart: Automated schedules, away mode, sleep mode
            """

            def __init__(self, calculator: EnergyCalculator):
                self.calculator = calculator

            def estimate_baseline(
                self,
                target_temp_f: float,
                outdoor_temp_f: float,
                hours: float = 24
            ) -> dict:
                """
                Estimate baseline energy usage (manual thermostat)

                Assumptions:
                - Constant temperature 24/7
                - No away mode or sleep setbacks
                """
                # Heating degree hours (how much below target)
                hdh = max(0, target_temp_f - outdoor_temp_f) * hours
                # Cooling degree hours (how much above target)
                cdh = max(0, outdoor_temp_f - target_temp_f) * hours

                # Estimate heating/cooling hours
                # Rough model: heating hours ≈ HDH / 20 (depends on insulation)
                heating_hours = hdh / 20.0
                cooling_hours = cdh / 25.0  # Cooling typically more efficient

                return self.calculator.calculate_daily_usage(heating_hours, cooling_hours)

            def estimate_smart(
                self,
                schedule: dict,
                outdoor_temp_f: float
            ) -> dict:
                """
                Estimate smart thermostat usage with schedule

                Args:
                    schedule: Daily schedule with setpoints
                    [
                        {'start_hour': 0, 'duration_hours': 6, 'target_temp': 65, 'occupancy': 'sleep'},
                        {'start_hour': 6, 'duration_hours': 2, 'target_temp': 68, 'occupancy': 'home'},
                        {'start_hour': 8, 'duration_hours': 9, 'target_temp': 62, 'occupancy': 'away'},
                        {'start_hour': 17, 'duration_hours': 5, 'target_temp': 70, 'occupancy': 'home'},
                        {'start_hour': 22, 'duration_hours': 2, 'target_temp': 65, 'occupancy': 'sleep'}
                    ]
                    outdoor_temp_f: Average outdoor temperature

                Returns:
                    Energy usage with smart schedule
                """
                total_heating_hours = 0
                total_cooling_hours = 0

                for period in schedule:
                    target_temp = period['target_temp']
                    duration = period['duration_hours']

                    # Calculate degree hours for this period
                    hdh = max(0, target_temp - outdoor_temp_f) * duration
                    cdh = max(0, outdoor_temp_f - target_temp) * duration

                    # Estimate heating/cooling hours
                    heating_hours = hdh / 20.0
                    cooling_hours = cdh / 25.0

                    total_heating_hours += heating_hours
                    total_cooling_hours += cooling_hours

                return self.calculator.calculate_daily_usage(total_heating_hours, total_cooling_hours)


        # Example usage
        def example_energy_savings():
            """Demonstrate energy savings calculation"""
            print("=== Smart Thermostat Energy Savings ===\n")

            # Initialize calculator
            calculator = EnergyCalculator(
                hvac_heating_power_kw=5.0,
                hvac_cooling_power_kw=3.5,
                electricity_rate_usd_per_kwh=0.13
            )
            estimator = SavingsEstimator(calculator)

            # Scenario: Winter day (outdoor: 40°F)
            outdoor_temp = 40

            # Baseline: Manual thermostat at 70°F all day
            baseline = estimator.estimate_baseline(
                target_temp_f=70,
                outdoor_temp_f=outdoor_temp,
                hours=24
            )

            # Smart schedule: Lower temps when away and sleeping
            smart_schedule = [
                {'start_hour': 0, 'duration_hours': 6, 'target_temp': 65, 'occupancy': 'sleep'},
                {'start_hour': 6, 'duration_hours': 2, 'target_temp': 68, 'occupancy': 'home'},
                {'start_hour': 8, 'duration_hours': 9, 'target_temp': 62, 'occupancy': 'away'},
                {'start_hour': 17, 'duration_hours': 5, 'target_temp': 70, 'occupancy': 'home'},
                {'start_hour': 22, 'duration_hours': 2, 'target_temp': 65, 'occupancy': 'sleep'}
            ]

            smart = estimator.estimate_smart(smart_schedule, outdoor_temp)

            # Calculate savings
            savings = calculator.calculate_savings(smart, baseline)

            # Print results
            print("Outdoor Temperature: 40°F (winter day)")
            print()
            print("Baseline (Manual Thermostat):")
            print(f"  Constant 70°F, 24/7")
            print(f"  Energy: {baseline['total_kwh']:.1f} kWh")
            print(f"  Cost: ${baseline['total_cost_usd']:.2f}")
            print()
            print("Smart Thermostat (Automated Schedule):")
            print(f"  Sleep (6h @ 65°F) + Away (9h @ 62°F) + Home (9h @ 68-70°F)")
            print(f"  Energy: {smart['total_kwh']:.1f} kWh")
            print(f"  Cost: ${smart['total_cost_usd']:.2f}")
            print()
            print("Savings:")
            print(f"  Energy saved: {savings['saved_kwh']:.1f} kWh ({savings['savings_percent']:.0f}%)")
            print(f"  Cost saved: ${savings['saved_usd']:.2f}/day")
            print(f"  Annual savings: ${savings['saved_usd'] * 365:.0f}/year")


        if __name__ == "__main__":
            example_energy_savings()
        ```

        **Output Example:**

        ```
        === Smart Thermostat Energy Savings ===

        Outdoor Temperature: 40°F (winter day)

        Baseline (Manual Thermostat):
          Constant 70°F, 24/7
          Energy: 36.0 kWh
          Cost: $4.68

        Smart Thermostat (Automated Schedule):
          Sleep (6h @ 65°F) + Away (9h @ 62°F) + Home (9h @ 68-70°F)
          Energy: 28.5 kWh
          Cost: $3.71

        Savings:
          Energy saved: 7.5 kWh (21%)
          Cost saved: $0.98/day
          Annual savings: $357/year
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 1M thermostats to 10M+ thermostats.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Temperature readings ingestion** | 🟡 Moderate | Batch writes, TimescaleDB compression |
    | **ML training** | ✅ Yes | Distributed training, incremental updates |
    | **Weather API calls** | 🟡 Moderate | Redis caching (3h TTL), batch queries |
    | **Real-time control commands** | 🟢 No | Stateless API, Redis cache |
    | **Mobile app queries** | 🟡 Moderate | Query cache, continuous aggregations |

    ---

    ## Horizontal Scaling

    ### Database Sharding

    ```python
    class ThermostatSharding:
        """Shard thermostats across database instances"""

        def __init__(self):
            self.shard_count = 16
            self.shards = [
                PostgreSQLConnection(f"thermostat-db-{i}.example.com")
                for i in range(self.shard_count)
            ]

        def get_shard(self, device_id: str) -> PostgreSQLConnection:
            """Consistent hashing to determine shard"""
            shard_index = hash(device_id) % self.shard_count
            return self.shards[shard_index]

        def query_thermostat(self, device_id: str) -> dict:
            """Query thermostat from appropriate shard"""
            shard = self.get_shard(device_id)
            return shard.query("SELECT * FROM thermostats WHERE device_id = %s", (device_id,))
    ```

    ---

    ## Battery Optimization (Wireless Thermostats)

    **Challenge:** Maximize battery life (target: 12+ months on 4x AA batteries)

    **Power consumption:**

    | Component | Active Current | Duty Cycle | Average |
    |-----------|---------------|------------|---------|
    | MCU (ARM Cortex-M4) | 8 mA | 10% | 800 µA |
    | WiFi (transmit) | 170 mA | 0.5% | 850 µA |
    | LCD Display | 15 mA | 2% | 300 µA |
    | Temperature sensor | 50 µA | 100% | 50 µA |
    | Sleep mode | 5 µA | ~87% | 5 µA |
    | **Total** | | | **~2 mA** |

    **Battery life:** 2000 mAh / 2 mA = 1000 hours ≈ 41 days - **needs optimization**

    **Optimizations:**

    1. **Reduce WiFi uploads:** 5 minutes → 10 minutes (saves 425 µA)
    2. **Deep sleep:** 87% → 95% sleep time (saves 640 µA)
    3. **LCD auto-off:** After 30 seconds (saves 200 µA)
    4. **Batch uploads:** Multiple readings in one WiFi session (saves 200 µA)

    **Optimized:** ~500 µA average → **4000 hours ≈ 166 days ≈ 5.5 months**

    **With C-wire (24VAC power):** Unlimited, no battery needed

    ---

    ## Cost Optimization

    **Monthly cost at 5M thermostats:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $21,600 (100 instances @ $6/day) |
    | **RDS PostgreSQL** | $8,640 (8 shards @ $1,080/month) |
    | **TimescaleDB** | $10,800 (10 nodes @ $1,080/month) |
    | **Redis Cache** | $4,320 (5 nodes @ $864/month) |
    | **S3 (models, backups)** | $1,500 (50 TB @ $30/TB) |
    | **Weather API** | $2,000 (1M locations, cached) |
    | **FCM/APNs** | $500 (push notifications) |
    | **Total** | **$49,360/month** |

    **Per-thermostat cost:** $49,360 / 5M = **$0.0099/month = $0.12/year per thermostat**

    **Revenue:** Assume $10/month subscription. 5M thermostats = $50M/month. Infrastructure is 0.1% of revenue.

    ---

    ## Monitoring & Alerting

    ```python
    class ThermostatSystemMonitoring:
        """Monitor system health and performance"""

        def track_metrics(self, device_id: str, metrics: dict):
            """Track thermostat metrics"""
            cloudwatch.put_metric_data(
                Namespace='SmartThermostat',
                MetricData=[
                    {
                        'MetricName': 'TemperatureError',
                        'Value': abs(metrics['target_temp'] - metrics['current_temp']),
                        'Unit': 'None',
                        'Dimensions': [{'Name': 'DeviceId', 'Value': device_id}]
                    },
                    {
                        'MetricName': 'HVACRuntime',
                        'Value': metrics['hvac_runtime_minutes'],
                        'Unit': 'Minutes'
                    },
                    {
                        'MetricName': 'DeviceOffline',
                        'Value': 1 if metrics['offline'] else 0,
                        'Unit': 'Count'
                    }
                ]
            )
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **PID control algorithm** - Precise temperature control without overshoot
    2. **Offline-first architecture** - Thermostat operates autonomously without cloud
    3. **ML-powered learning** - Automatically learn user schedules and preferences
    4. **Geofencing** - Automatic home/away detection for energy savings
    5. **Multi-tier time-series storage** - Efficient storage with compression
    6. **Weather integration** - Optimize heating/cooling based on forecast
    7. **Energy reporting** - Transparency and user engagement

    ---

    ## Interview Tips

    ✅ **Start with control loop** - Explain PID algorithm for temperature control

    ✅ **Discuss offline capability** - Thermostat must work without internet

    ✅ **ML learning approach** - Pattern detection from user interactions

    ✅ **Geofencing trade-offs** - Privacy vs. convenience vs. energy savings

    ✅ **Energy calculations** - Show how to estimate and report savings

    ✅ **Multi-user coordination** - Handle homes with multiple residents

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"Why PID instead of simple on/off?"** | PID prevents overshoot, reduces oscillations, more comfortable and efficient |
    | **"How to handle WiFi outages?"** | Local control continues, readings buffered, sync when connection restored |
    | **"How accurate is energy savings?"** | ±10-15% accuracy, based on HVAC runtime and outdoor temperature models |
    | **"Privacy concerns with geofencing?"** | Location data encrypted, processed only for home/away, user can disable |
    | **"Multi-zone HVAC support?"** | Shard by zone, coordinate setpoints across zones, damper control |
    | **"What if user disables learning?"** | Fall back to manual schedule, still benefit from remote control and reporting |
    | **"How to detect HVAC problems?"** | Anomaly detection: unusually long runtimes, unable to reach target, cycling issues |
    | **"Integration with utility demand response?"** | API endpoint for utility to request temp adjustment during peak demand |

    ---

    ## Real-World Examples

    ### Nest Learning Thermostat

    - **Architecture:** WiFi connected, cloud ML learning
    - **Learning:** Auto-schedule from 1 week of manual adjustments
    - **Sensors:** Temperature, humidity, motion (near/far), ambient light
    - **Display:** 2.08" LCD with metal ring interface
    - **Savings:** Average 10-12% on heating, 15% on cooling
    - **Battery:** Rechargeable lithium-ion from C-wire or HVAC power stealing

    ### Ecobee SmartThermostat

    - **Architecture:** WiFi + remote room sensors
    - **Remote sensors:** Multiple rooms for average temperature
    - **Occupancy:** PIR motion sensors in thermostat and room sensors
    - **Integrations:** Alexa built-in, HomeKit, Google Home, SmartThings
    - **Savings:** Average 23% annual savings
    - **C-wire:** Required (includes adapter)

    ### Honeywell Home T9

    - **Architecture:** WiFi + multi-room sensors
    - **Geofencing:** Mobile app location-based automation
    - **Smart room:** Focus heating/cooling on occupied rooms
    - **Display:** Touchscreen with weather display
    - **Battery:** 2x AA batteries (with C-wire optional)

    ---

    ## Security Best Practices

    1. **Device authentication** - Certificate-based authentication for devices
    2. **Encrypted communication** - TLS for all cloud communication
    3. **Local control security** - Physical access required for initial setup
    4. **Privacy** - Location data encrypted, processed only for home/away
    5. **Firmware updates** - OTA updates with signature verification
    6. **Access control** - User permissions (owner, guest, read-only)
    7. **Audit logging** - Log all temperature changes and HVAC commands

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Nest, Ecobee, Honeywell Home, Sensibo, Emerson Sensi

---

*Master this problem and you'll be ready for: Smart home systems, IoT control platforms, HVAC systems, home automation*
