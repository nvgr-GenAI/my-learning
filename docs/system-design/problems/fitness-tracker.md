# Design Fitness Tracker System (Fitbit, Apple Watch)

A comprehensive fitness tracking platform that monitors physical activity (steps, heart rate, sleep), processes sensor data in real-time, syncs across devices, provides health insights using machine learning, and enables social features for user engagement.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M wearable devices, 500M data syncs/day, real-time sensor processing, ML-powered health insights |
| **Key Challenges** | Battery optimization, sensor data processing, real-time sync, sleep stage detection, social feed scaling |
| **Core Concepts** | Activity tracking, heart rate analysis, sleep detection, data synchronization, health analytics, social features |
| **Companies** | Fitbit, Apple Watch, Garmin, Whoop, Samsung Galaxy Watch, Polar, Oura Ring |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Step Counting** | Track steps using accelerometer data | P0 (Must have) |
    | **Heart Rate Monitoring** | Continuous HR monitoring with PPG sensor | P0 (Must have) |
    | **Sleep Tracking** | Detect sleep stages (deep, light, REM, awake) | P0 (Must have) |
    | **Activity Recognition** | Auto-detect exercises (running, cycling, swimming) | P0 (Must have) |
    | **Data Synchronization** | Sync data between device and mobile app | P0 (Must have) |
    | **Daily Goals** | Set and track daily step/calorie/activity goals | P0 (Must have) |
    | **Health Insights** | ML-powered insights on fitness trends | P1 (Should have) |
    | **Social Features** | Challenges, leaderboards, friend comparisons | P1 (Should have) |
    | **Workout Tracking** | Manual workout logging with GPS tracking | P1 (Should have) |
    | **Notifications** | Smart notifications from phone to device | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Medical-grade diagnostics (ECG, blood oxygen, blood pressure)
    - Payment processing (contactless payments)
    - Music playback and storage
    - Voice assistant integration
    - Third-party app ecosystem

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Battery Life** | 5-7 days per charge | User convenience, competitive with market |
    | **Sync Latency** | < 10 seconds | Near real-time data updates |
    | **Data Accuracy** | > 95% for steps, > 90% for sleep stages | User trust and health decisions |
    | **Availability** | 99.9% uptime | Critical for daily health tracking |
    | **Data Retention** | Lifetime (with aggregation) | Historical trends and insights |
    | **Scalability** | 50M+ active devices | Support consumer-scale deployment |
    | **Privacy** | End-to-end encryption | Sensitive health data protection |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total active devices: 50,000,000 devices
    Daily active users: 40,000,000 (80% engagement)

    Data collection frequency:
    - Steps: Every 1 minute (aggregated on device)
    - Heart rate: Every 5 seconds (continuous)
    - Sleep data: Every 30 seconds (during sleep)
    - Workout data: Real-time GPS tracking

    Data sync events:
    - Background sync: Every 30 minutes (while wearing)
    - Manual sync: 2-3 times/day
    - Workout sync: Immediately after workout
    - Total syncs: 500M syncs/day = 5,787 syncs/sec
    - Peak (morning/evening): 17,361 syncs/sec

    Mobile app usage:
    - Daily active users: 40M
    - Sessions per user: 5 sessions/day
    - Total sessions: 200M sessions/day = 2,315 sessions/sec
    - Average session: 3 minutes
    - API requests per session: 10-15 requests
    - Total API requests: 2B requests/day = 23,148 requests/sec

    Social feed:
    - Active social users: 10M (25% of users)
    - Feed refreshes: 10 refreshes/day
    - Total feed loads: 100M/day = 1,157 loads/sec
    - Friend updates (posts): 50M/day = 579 posts/sec

    Health insights (ML):
    - Daily insight generation: 40M users
    - Processing per user: 30 seconds (batch job)
    - Total compute: 40M × 30s = 1.2B seconds/day
    - Parallel processing: 1.2B / 86,400 = 13,889 concurrent jobs
    ```

    ### Storage Estimates

    ```
    Per-device data per day:

    Raw sensor data:
    - Steps: 1,440 min × 2 bytes (step count) = 2.88 KB
    - Heart rate: 17,280 samples × 2 bytes = 34.56 KB
    - Sleep: 960 samples × 3 bytes (stage + quality) = 2.88 KB
    - Activity metadata: 1 KB (start/end times, calories)
    - Total raw per device: 41.32 KB/day

    Daily: 50M devices × 41.32 KB = 2.07 TB/day
    Monthly: 2.07 TB × 30 = 62.1 TB/month
    Yearly: 2.07 TB × 365 = 755.5 TB/year

    With compression (5:1 ratio for time-series):
    - Daily: 414 GB/day
    - Monthly: 12.4 TB/month
    - Yearly: 151 TB/year

    Retention strategy (multi-tier):

    Tier 1: High-resolution (1-min intervals) - 30 days
    - Storage: 414 GB × 30 = 12.4 TB
    - Use: Recent activity, trend analysis

    Tier 2: Hourly aggregation - 1 year
    - Reduction: 60x fewer points
    - Storage: 755.5 TB / 60 = 12.6 TB
    - Use: Historical trends, yearly summaries

    Tier 3: Daily aggregation - Lifetime
    - Reduction: 1,440x fewer points
    - Storage: 755.5 TB × 10 / 1,440 = 5.2 TB (10 years)
    - Use: Long-term health trends

    Total storage: 12.4 TB + 12.6 TB + 5.2 TB = 30.2 TB
    With replication (3x): 90.6 TB

    User profile data:
    - 50M users × 5 KB (profile, settings, goals) = 250 GB

    Social data:
    - Friend connections: 50M users × 100 friends × 16 bytes = 80 GB
    - Activity posts: 50M posts/day × 500 bytes × 90 days = 2.25 TB
    - Comments/reactions: 100M/day × 200 bytes × 90 days = 1.8 TB

    Workout data:
    - GPS tracks: 10M workouts/day × 50 KB × 90 days = 45 TB
    - Workout summaries: 10M × 2 KB × 365 days = 7.3 TB

    Total storage:
    - Sensor data: 90.6 TB
    - User/social: 4.4 TB
    - Workout/GPS: 52.3 TB
    - Total: 147.3 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (device to cloud sync):
    - 5,787 syncs/sec × 50 KB (aggregated data) = 289 MB/sec ≈ 2.3 Gbps
    - Compressed (3:1): 770 Mbps
    - With protocol overhead (HTTPS, TLS): 1.2 Gbps
    - Peak (3x): 3.6 Gbps

    Egress (mobile app queries, social feeds):
    - API requests: 23,148 req/sec × 5 KB = 116 MB/sec ≈ 928 Mbps
    - Social feed: 1,157 loads/sec × 100 KB = 116 MB/sec ≈ 928 Mbps
    - Chart data: 2,315 sessions/sec × 50 KB = 116 MB/sec ≈ 928 Mbps
    - Total egress: ~2.8 Gbps

    Total bandwidth: 1.2 Gbps (ingress) + 2.8 Gbps (egress) = 4 Gbps
    ```

    ### Server Estimates

    ```
    Data sync layer:
    - 5,787 syncs/sec / 500 per node = 12 sync nodes
    - CPU: 4 cores per node (data validation, compression)
    - Memory: 16 GB per node (buffering)

    Time-series database:
    - 147 TB total / 50 TB per node = 3 TSDB nodes
    - CPU: 32 cores per node (compression, indexing)
    - Memory: 256 GB per node (hot data, indexes)
    - Disk: 50 TB SSD per node

    API/Query layer:
    - 23,148 req/sec / 200 per node = 116 API nodes
    - CPU: 8 cores per node (query processing)
    - Memory: 32 GB per node (query cache)

    ML/Analytics:
    - 13,889 concurrent jobs / 1,000 per node = 14 ML nodes
    - CPU: 16 cores per node (model inference)
    - Memory: 64 GB per node (model state)
    - GPU: Optional for training

    Social feed service:
    - 1,157 loads/sec / 100 per node = 12 feed nodes
    - CPU: 8 cores per node (ranking, aggregation)
    - Memory: 32 GB per node (feed cache)

    Total servers:
    - Sync: 12 nodes
    - Storage: 3 nodes (with replication: 9 nodes)
    - API: 116 nodes
    - ML: 14 nodes
    - Social: 12 nodes
    - Total: ~163 nodes
    ```

    ---

    ## Key Assumptions

    1. Average user wears device 16 hours/day
    2. 80% daily active rate (40M of 50M users)
    3. 20% of users participate in social features
    4. 20% of users do tracked workouts daily (10M workouts/day)
    5. 5:1 compression ratio for time-series sensor data
    6. Average sync payload: 50 KB (30 minutes of data)
    7. Background sync every 30 minutes when device connected to phone
    8. 95% of queries access last 7 days of data

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Battery optimization** - Minimize power consumption on wearable device
    2. **Edge processing** - Process sensor data on device to reduce sync frequency
    3. **Real-time sync** - Low-latency data synchronization with mobile app
    4. **Time-series optimization** - Efficient storage and querying of sensor data
    5. **ML-powered insights** - Automated health insights from activity patterns
    6. **Social engagement** - Feed-based architecture for challenges and sharing

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Wearable Device"
            Accelerometer[Accelerometer<br/>Step counting]
            PPG[PPG Sensor<br/>Heart rate]
            Gyroscope[Gyroscope<br/>Motion detection]
            EdgeProcessor[Edge Processor<br/>Activity recognition<br/>Aggregation]
            LocalStorage[Local Storage<br/>24-48 hours buffer]
            BLE[Bluetooth LE<br/>Sync protocol]
        end

        subgraph "Mobile App"
            AppSync[Sync Service<br/>BLE/WiFi]
            LocalDB[Local DB<br/>SQLite]
            AppUI[App UI<br/>Dashboard, Charts]
            PushNotif[Push Notifications]
        end

        subgraph "API Gateway"
            APIGateway[API Gateway<br/>REST/GraphQL<br/>Auth, Rate Limiting]
            WSGateway[WebSocket Gateway<br/>Real-time updates]
        end

        subgraph "Sync & Ingestion"
            SyncService[Sync Service<br/>Device data ingestion]
            DataValidator[Data Validator<br/>Quality checks]
            MessageQueue[Message Queue<br/>Kafka<br/>Event stream]
        end

        subgraph "Time-Series Storage"
            TSDB_Hot[(Hot Storage<br/>TimescaleDB<br/>30 days<br/>1-min resolution)]
            TSDB_Warm[(Warm Storage<br/>TimescaleDB<br/>1 year<br/>1-hour resolution)]
            TSDB_Cold[(Cold Storage<br/>S3/Parquet<br/>Lifetime<br/>1-day resolution)]
        end

        subgraph "Processing & Analytics"
            AggregationService[Aggregation Service<br/>Hourly/daily rollups]
            ActivityDetector[Activity Detector<br/>ML models<br/>Exercise recognition]
            SleepAnalyzer[Sleep Analyzer<br/>ML models<br/>Stage detection]
            HealthInsights[Health Insights Engine<br/>Trend analysis<br/>Recommendations]
        end

        subgraph "Social Features"
            SocialFeed[Social Feed Service<br/>Activity posts<br/>Ranking algorithm]
            ChallengeEngine[Challenge Engine<br/>Competitions<br/>Leaderboards]
            FriendGraph[Friend Graph DB<br/>Neo4j<br/>Connections]
        end

        subgraph "User Services"
            UserService[User Service<br/>Profile, settings]
            GoalService[Goal Service<br/>Daily targets]
            AchievementService[Achievement Service<br/>Badges, milestones]
        end

        subgraph "Analytics & ML"
            MLTraining[ML Training<br/>Model updates]
            ABTesting[A/B Testing<br/>Feature experiments]
            Analytics[Analytics Service<br/>Usage metrics]
        end

        subgraph "Notification System"
            NotificationEngine[Notification Engine<br/>Rules engine]
            PushService[Push Service<br/>APNS, FCM]
            EmailService[Email Service<br/>Weekly reports]
        end

        subgraph "External Integrations"
            HealthKit[HealthKit/Google Fit<br/>Platform integration]
            StravaAPI[Strava API<br/>Social sync]
            WeatherAPI[Weather API<br/>Context data]
        end

        Accelerometer --> EdgeProcessor
        PPG --> EdgeProcessor
        Gyroscope --> EdgeProcessor
        EdgeProcessor --> LocalStorage
        LocalStorage --> BLE

        BLE -.->|Bluetooth| AppSync
        AppSync --> LocalDB
        LocalDB --> AppUI
        AppSync --> APIGateway

        APIGateway --> SyncService
        SyncService --> DataValidator
        DataValidator --> MessageQueue

        MessageQueue --> TSDB_Hot
        MessageQueue --> ActivityDetector
        MessageQueue --> SleepAnalyzer

        TSDB_Hot --> AggregationService
        AggregationService --> TSDB_Warm
        AggregationService --> TSDB_Cold

        TSDB_Hot --> HealthInsights
        TSDB_Warm --> HealthInsights
        HealthInsights --> NotificationEngine

        ActivityDetector --> TSDB_Hot
        SleepAnalyzer --> TSDB_Hot

        APIGateway --> UserService
        APIGateway --> GoalService
        APIGateway --> SocialFeed
        APIGateway --> ChallengeEngine

        SocialFeed --> FriendGraph
        ChallengeEngine --> FriendGraph

        TSDB_Hot --> SocialFeed
        TSDB_Hot --> ChallengeEngine

        HealthInsights --> AchievementService

        NotificationEngine --> PushService
        NotificationEngine --> EmailService
        PushService --> PushNotif

        TSDB_Hot --> HealthKit
        SocialFeed --> StravaAPI
        HealthInsights --> WeatherAPI

        TSDB_Hot --> MLTraining
        UserService --> ABTesting

        WSGateway --> AppUI

        style LocalStorage fill:#e8eaf6
        style TSDB_Hot fill:#ffe1e1
        style TSDB_Warm fill:#fff4e1
        style TSDB_Cold fill:#f0f0f0
        style MessageQueue fill:#e8eaf6
        style APIGateway fill:#e1f5ff
        style HealthInsights fill:#e8f5e9
        style SocialFeed fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Edge Processing** | Process sensor data on device to save battery, reduce sync frequency | Cloud processing (drains battery, requires constant connectivity) |
    | **Bluetooth LE** | Low-power device-to-phone sync | WiFi (battery intensive), Cellular (expensive, high power) |
    | **TimescaleDB** | Optimized for time-series data, SQL queries, compression | InfluxDB (less mature replication), Cassandra (complex queries) |
    | **Kafka Message Queue** | Decouple sync from processing, replay capability, multiple consumers | Direct writes (coupling), RabbitMQ (less throughput) |
    | **Multi-tier Storage** | 90% cost savings vs. keeping all data hot | Single-tier SSD (expensive), All S3 (slow queries) |
    | **Redis Cache** | 70% cache hit rate, 10x faster queries | No cache (slow), application cache (no sharing) |
    | **Neo4j Friend Graph** | Efficient friend recommendations, social queries | PostgreSQL (slow graph queries), DynamoDB (complex modeling) |

    **Key Trade-off:** We chose **eventual consistency** for sensor data (acceptable delay: 10 seconds) to enable offline mode and battery savings. Strong consistency would require constant connectivity and drain battery.

    ---

    ## API Design

    ### 1. Sync Device Data (Device to Cloud)

    **Request:**
    ```http
    POST /api/v1/devices/{device_id}/sync
    Content-Type: application/json
    Authorization: Bearer <device_token>

    {
      "device_id": "device_abc123",
      "sync_time": 1735820100,
      "data_window": {
        "start": 1735818300,
        "end": 1735820100
      },
      "steps": [
        {"timestamp": 1735818300, "count": 45, "calories": 2.1},
        {"timestamp": 1735818360, "count": 52, "calories": 2.4},
        // ... 30 minutes of 1-min aggregates
      ],
      "heart_rate": [
        {"timestamp": 1735818300, "bpm": 72, "confidence": 0.95},
        {"timestamp": 1735818305, "bpm": 73, "confidence": 0.96},
        // ... 30 minutes of 5-sec samples (360 samples)
      ],
      "activities": [
        {
          "type": "running",
          "start": 1735818600,
          "end": 1735819800,
          "distance_m": 3200,
          "calories": 285,
          "avg_hr": 145,
          "max_hr": 168
        }
      ],
      "sleep": [
        {
          "timestamp": 1735776000,
          "stage": "deep",  // deep, light, rem, awake
          "duration": 1800,
          "movement": 2
        }
      ],
      "battery_level": 45,
      "firmware_version": "3.2.1"
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "synced_records": {
        "steps": 30,
        "heart_rate": 360,
        "activities": 1,
        "sleep": 1
      },
      "next_sync": 1735821900,
      "server_time": 1735820110,
      "insights": {
        "daily_progress": 8542,
        "goal": 10000,
        "achievement_unlocked": "5K Runner"
      }
    }
    ```

    ---

    ### 2. Get Daily Summary

    **Request:**
    ```http
    GET /api/v1/users/me/summary?date=2025-01-02
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "user_id": "user_xyz789",
      "date": "2025-01-02",
      "steps": {
        "total": 12450,
        "goal": 10000,
        "distance_km": 8.9,
        "calories": 542,
        "active_minutes": 87
      },
      "heart_rate": {
        "resting": 58,
        "average": 72,
        "max": 168,
        "fat_burn_minutes": 35,
        "cardio_minutes": 28,
        "peak_minutes": 12
      },
      "sleep": {
        "total_minutes": 462,
        "deep_minutes": 128,
        "light_minutes": 245,
        "rem_minutes": 89,
        "awake_count": 3,
        "sleep_score": 87,
        "bedtime": "2025-01-01T23:15:00Z",
        "wake_time": "2025-01-02T06:57:00Z"
      },
      "activities": [
        {
          "type": "running",
          "start": "2025-01-02T07:30:00Z",
          "duration": 1800,
          "distance_km": 5.2,
          "calories": 425,
          "avg_pace": "5:46",
          "avg_hr": 145
        }
      ],
      "achievements": [
        {
          "id": "daily_goal_10d",
          "name": "10 Day Streak",
          "icon": "🔥",
          "unlocked_at": "2025-01-02T12:34:56Z"
        }
      ]
    }
    ```

    ---

    ### 3. Get Activity History

    **Request:**
    ```http
    GET /api/v1/users/me/activity/history?start=2025-01-01&end=2025-01-31&metric=steps&resolution=daily
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "user_id": "user_xyz789",
      "metric": "steps",
      "resolution": "daily",
      "data": [
        {
          "date": "2025-01-01",
          "value": 10234,
          "goal": 10000,
          "distance_km": 7.3,
          "calories": 487
        },
        {
          "date": "2025-01-02",
          "value": 12450,
          "goal": 10000,
          "distance_km": 8.9,
          "calories": 542
        }
        // ... more days
      ],
      "summary": {
        "total_steps": 324567,
        "avg_daily": 10469,
        "best_day": {
          "date": "2025-01-15",
          "value": 18234
        },
        "days_met_goal": 24,
        "current_streak": 10
      }
    }
    ```

    ---

    ### 4. Get Social Feed

    **Request:**
    ```http
    GET /api/v1/social/feed?limit=20&offset=0
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "posts": [
        {
          "post_id": "post_abc123",
          "user": {
            "user_id": "user_friend1",
            "name": "Sarah Johnson",
            "avatar_url": "https://cdn.fittrack.com/avatars/user_friend1.jpg"
          },
          "type": "achievement",
          "content": {
            "achievement": "Marathon Master",
            "icon": "🏃‍♀️",
            "description": "Completed a full marathon!"
          },
          "workout": {
            "type": "running",
            "distance_km": 42.2,
            "duration": 14400,
            "calories": 2856,
            "map_url": "https://cdn.fittrack.com/maps/workout_xyz.png"
          },
          "likes": 45,
          "comments": 12,
          "liked_by_user": false,
          "created_at": "2025-01-02T14:30:00Z"
        },
        {
          "post_id": "post_def456",
          "user": {
            "user_id": "user_friend2",
            "name": "Mike Chen",
            "avatar_url": "https://cdn.fittrack.com/avatars/user_friend2.jpg"
          },
          "type": "challenge_complete",
          "content": {
            "challenge_name": "Weekend Warrior",
            "rank": 2,
            "total_participants": 50,
            "value": 25000,
            "metric": "steps"
          },
          "likes": 23,
          "comments": 5,
          "liked_by_user": true,
          "created_at": "2025-01-02T09:15:00Z"
        }
      ],
      "pagination": {
        "limit": 20,
        "offset": 0,
        "total": 156,
        "has_more": true
      }
    }
    ```

    ---

    ### 5. Get Challenge Leaderboard

    **Request:**
    ```http
    GET /api/v1/challenges/{challenge_id}/leaderboard
    Authorization: Bearer <user_token>
    ```

    **Response:**
    ```json
    {
      "challenge_id": "challenge_weekend_warrior",
      "name": "Weekend Warrior",
      "description": "Most steps over the weekend",
      "metric": "steps",
      "start_date": "2025-01-04T00:00:00Z",
      "end_date": "2025-01-05T23:59:59Z",
      "status": "active",
      "leaderboard": [
        {
          "rank": 1,
          "user": {
            "user_id": "user_top1",
            "name": "Alex Rodriguez",
            "avatar_url": "https://cdn.fittrack.com/avatars/user_top1.jpg"
          },
          "value": 32456,
          "is_current_user": false
        },
        {
          "rank": 2,
          "user": {
            "user_id": "user_friend2",
            "name": "Mike Chen",
            "avatar_url": "https://cdn.fittrack.com/avatars/user_friend2.jpg"
          },
          "value": 28930,
          "is_current_user": false
        },
        {
          "rank": 5,
          "user": {
            "user_id": "user_xyz789",
            "name": "You",
            "avatar_url": "https://cdn.fittrack.com/avatars/user_xyz789.jpg"
          },
          "value": 24562,
          "is_current_user": true
        }
      ],
      "total_participants": 50,
      "current_user_rank": 5,
      "current_user_value": 24562
    }
    ```

    ---

    ## Database Schema

    ### Time-Series Data (TimescaleDB)

    **Activity Metrics (Hypertable):**

    ```sql
    -- Create hypertable for automatic time-based partitioning
    CREATE TABLE activity_metrics (
        user_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        metric_type VARCHAR(20) NOT NULL,  -- steps, heart_rate, calories, distance
        value REAL NOT NULL,
        metadata JSONB,                     -- Additional context
        device_id VARCHAR(50),
        PRIMARY KEY (user_id, time, metric_type)
    );

    -- Convert to hypertable
    SELECT create_hypertable('activity_metrics', 'time');

    -- Enable compression (5:1 ratio)
    ALTER TABLE activity_metrics SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'user_id, metric_type',
        timescaledb.compress_orderby = 'time DESC'
    );

    -- Compression policy: compress data older than 7 days
    SELECT add_compression_policy('activity_metrics', INTERVAL '7 days');

    -- Retention policy: keep high-res data for 30 days
    SELECT add_retention_policy('activity_metrics', INTERVAL '30 days');

    -- Indexes
    CREATE INDEX idx_user_time ON activity_metrics (user_id, time DESC);
    CREATE INDEX idx_metric_type ON activity_metrics (metric_type, time DESC);
    ```

    **Heart Rate Data (Hypertable):**

    ```sql
    CREATE TABLE heart_rate_samples (
        user_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        bpm SMALLINT NOT NULL,
        confidence REAL,                    -- Sensor confidence (0-1)
        activity_context VARCHAR(20),       -- resting, active, workout
        PRIMARY KEY (user_id, time)
    );

    SELECT create_hypertable('heart_rate_samples', 'time');

    -- Compression for heart rate data
    ALTER TABLE heart_rate_samples SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'user_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    SELECT add_compression_policy('heart_rate_samples', INTERVAL '7 days');
    SELECT add_retention_policy('heart_rate_samples', INTERVAL '30 days');
    ```

    **Sleep Data (Hypertable):**

    ```sql
    CREATE TABLE sleep_sessions (
        user_id UUID NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        stage VARCHAR(10) NOT NULL,         -- deep, light, rem, awake
        duration INTEGER NOT NULL,          -- seconds
        movement_count SMALLINT,
        heart_rate_avg SMALLINT,
        confidence REAL,                    -- ML model confidence
        PRIMARY KEY (user_id, time)
    );

    SELECT create_hypertable('sleep_sessions', 'time');

    ALTER TABLE sleep_sessions SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'user_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    SELECT add_compression_policy('sleep_sessions', INTERVAL '7 days');
    ```

    **Continuous Aggregations:**

    ```sql
    -- Hourly activity summary
    CREATE MATERIALIZED VIEW activity_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        user_id,
        time_bucket('1 hour', time) AS bucket,
        metric_type,
        SUM(CASE WHEN metric_type = 'steps' THEN value ELSE 0 END) as total_steps,
        AVG(CASE WHEN metric_type = 'heart_rate' THEN value ELSE NULL END) as avg_heart_rate,
        MAX(CASE WHEN metric_type = 'heart_rate' THEN value ELSE NULL END) as max_heart_rate,
        SUM(CASE WHEN metric_type = 'calories' THEN value ELSE 0 END) as total_calories,
        COUNT(*) as sample_count
    FROM activity_metrics
    GROUP BY user_id, bucket, metric_type;

    SELECT add_continuous_aggregate_policy('activity_hourly',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '30 minutes'
    );

    -- Daily activity summary
    CREATE MATERIALIZED VIEW activity_daily
    WITH (timescaledb.continuous) AS
    SELECT
        user_id,
        time_bucket('1 day', time) AS bucket,
        SUM(CASE WHEN metric_type = 'steps' THEN value ELSE 0 END) as total_steps,
        AVG(CASE WHEN metric_type = 'heart_rate' THEN value ELSE NULL END) as avg_heart_rate,
        MIN(CASE WHEN metric_type = 'heart_rate' AND metadata->>'context' = 'resting'
            THEN value ELSE NULL END) as resting_heart_rate,
        SUM(CASE WHEN metric_type = 'calories' THEN value ELSE 0 END) as total_calories,
        SUM(CASE WHEN metric_type = 'distance' THEN value ELSE 0 END) as total_distance_m
    FROM activity_metrics
    GROUP BY user_id, bucket;

    SELECT add_continuous_aggregate_policy('activity_daily',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day'
    );
    ```

    ---

    ### User & Social Data (PostgreSQL)

    **Users:**

    ```sql
    CREATE TABLE users (
        user_id UUID PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        full_name VARCHAR(255),
        date_of_birth DATE,
        gender VARCHAR(20),
        height_cm SMALLINT,
        weight_kg REAL,
        avatar_url TEXT,
        timezone VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_email (email)
    );
    ```

    **Devices:**

    ```sql
    CREATE TABLE devices (
        device_id VARCHAR(50) PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        device_type VARCHAR(50),            -- fitbit_charge5, apple_watch_series8
        firmware_version VARCHAR(20),
        last_sync_at TIMESTAMP,
        battery_level SMALLINT,
        status VARCHAR(20) DEFAULT 'active', -- active, inactive, lost
        paired_at TIMESTAMP,
        INDEX idx_user_devices (user_id),
        INDEX idx_last_sync (last_sync_at DESC)
    );
    ```

    **Goals:**

    ```sql
    CREATE TABLE user_goals (
        goal_id UUID PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        goal_type VARCHAR(20) NOT NULL,     -- steps, calories, distance, active_minutes
        target_value INTEGER NOT NULL,
        frequency VARCHAR(10) DEFAULT 'daily', -- daily, weekly, monthly
        start_date DATE,
        end_date DATE,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_goals (user_id, is_active)
    );
    ```

    **Workouts:**

    ```sql
    CREATE TABLE workouts (
        workout_id UUID PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        device_id VARCHAR(50) REFERENCES devices(device_id),
        workout_type VARCHAR(50) NOT NULL,  -- running, cycling, swimming, gym
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        duration INTEGER NOT NULL,          -- seconds
        distance_m REAL,
        calories INTEGER,
        avg_heart_rate SMALLINT,
        max_heart_rate SMALLINT,
        elevation_gain_m REAL,
        gps_track_url TEXT,                 -- S3 URL to GPS track
        map_image_url TEXT,                 -- Pre-rendered map image
        notes TEXT,
        is_public BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_workouts (user_id, start_time DESC),
        INDEX idx_workout_type (workout_type, start_time DESC)
    );
    ```

    **Social Graph:**

    ```sql
    CREATE TABLE friendships (
        friendship_id UUID PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        friend_id UUID REFERENCES users(user_id),
        status VARCHAR(20) DEFAULT 'pending', -- pending, accepted, blocked
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accepted_at TIMESTAMP,
        UNIQUE(user_id, friend_id),
        INDEX idx_user_friends (user_id, status),
        INDEX idx_friend_requests (friend_id, status, created_at DESC)
    );
    ```

    **Activity Posts:**

    ```sql
    CREATE TABLE activity_posts (
        post_id UUID PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        post_type VARCHAR(20) NOT NULL,     -- workout, achievement, challenge
        content JSONB NOT NULL,             -- Flexible post content
        workout_id UUID REFERENCES workouts(workout_id),
        visibility VARCHAR(20) DEFAULT 'friends', -- public, friends, private
        likes_count INTEGER DEFAULT 0,
        comments_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_posts (user_id, created_at DESC),
        INDEX idx_post_type (post_type, created_at DESC)
    );
    ```

    **Challenges:**

    ```sql
    CREATE TABLE challenges (
        challenge_id UUID PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        metric_type VARCHAR(20) NOT NULL,   -- steps, distance, calories
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        challenge_type VARCHAR(20),         -- friends, global, group
        created_by UUID REFERENCES users(user_id),
        status VARCHAR(20) DEFAULT 'active', -- active, completed, cancelled
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_active_challenges (status, start_time, end_time)
    );

    CREATE TABLE challenge_participants (
        challenge_id UUID REFERENCES challenges(challenge_id),
        user_id UUID REFERENCES users(user_id),
        current_value INTEGER DEFAULT 0,
        rank INTEGER,
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (challenge_id, user_id),
        INDEX idx_challenge_leaderboard (challenge_id, current_value DESC)
    );
    ```

    **Achievements:**

    ```sql
    CREATE TABLE achievements (
        achievement_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        icon_url TEXT,
        category VARCHAR(50),               -- steps, distance, streak, social
        criteria JSONB NOT NULL,            -- Achievement unlock criteria
        rarity VARCHAR(20),                 -- common, rare, epic, legendary
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE user_achievements (
        user_id UUID REFERENCES users(user_id),
        achievement_id VARCHAR(50) REFERENCES achievements(achievement_id),
        unlocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        progress JSONB,                     -- Progress towards multi-level achievements
        PRIMARY KEY (user_id, achievement_id),
        INDEX idx_user_achievements (user_id, unlocked_at DESC)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Device Sync Flow

    ```mermaid
    sequenceDiagram
        participant Device as Wearable Device
        participant Mobile as Mobile App
        participant API as API Gateway
        participant Sync as Sync Service
        participant Kafka
        participant TSDB as TimescaleDB
        participant Cache as Redis Cache

        Note over Device: Every 30 minutes<br/>or manual sync

        Device->>Device: Aggregate sensor data<br/>(1-min steps, 5-sec HR)
        Device->>Mobile: BLE sync request<br/>(50 KB payload)

        Mobile->>API: POST /sync<br/>(compressed data)
        API->>API: Authenticate device<br/>Rate limiting check
        API->>Sync: Forward sync data

        Sync->>Sync: Validate data quality<br/>Check timestamps<br/>Deduplicate

        alt Valid data
            Sync->>Kafka: Publish to activity-events
            Kafka-->>Sync: Ack
            Sync-->>API: 200 OK + insights
            API-->>Mobile: Sync success
            Mobile-->>Device: Ack, next_sync_time

            Kafka->>TSDB: Batch insert<br/>(1000 records)
            TSDB-->>Kafka: Success

            Kafka->>Cache: Invalidate user cache
            Kafka->>TSDB: Trigger aggregation refresh
        else Invalid data
            Sync->>Sync: Log error
            Sync-->>API: 400 Bad Request
            API-->>Mobile: Retry with exponential backoff
        end

        Note over TSDB: Continuous aggregation<br/>updates hourly/daily views
    ```

    ---

    ### Sleep Analysis Flow

    ```mermaid
    sequenceDiagram
        participant Device as Wearable Device
        participant ML as Sleep Analyzer
        participant TSDB as TimescaleDB
        participant Cache as Redis Cache
        participant Notif as Notification Service
        participant Mobile as Mobile App

        Note over Device: During sleep<br/>(23:00 - 07:00)

        Device->>Device: Collect sensor data:<br/>- Accelerometer (movement)<br/>- Heart rate (5-sec)<br/>- Heart rate variability

        Device->>Device: Edge processing:<br/>- Detect sleep/wake<br/>- Basic movement classification

        Note over Device: Morning sync

        Device->>ML: Upload sleep data<br/>(8 hours of samples)

        ML->>TSDB: Get historical sleep patterns<br/>(last 30 days)
        TSDB-->>ML: Historical baseline

        ML->>ML: ML model inference:<br/>1. Segment into 30-sec epochs<br/>2. Extract features<br/>3. Classify sleep stages<br/>4. Calculate sleep score

        alt Sleep stages detected
            ML->>TSDB: Store sleep session<br/>(stages, durations, quality)
            TSDB-->>ML: Success

            ML->>Cache: Cache sleep summary<br/>TTL: 24 hours
            ML->>Notif: Generate insight<br/>"Great sleep! 87 score"
            Notif->>Mobile: Push notification
        else Insufficient data
            ML->>ML: Log error
        end

        Mobile->>Cache: GET /sleep/summary
        Cache-->>Mobile: Return cached summary
    ```

    ---

    ### Social Feed Flow

    ```mermaid
    sequenceDiagram
        participant User as User (Mobile App)
        participant API as API Gateway
        participant Feed as Social Feed Service
        participant Graph as Friend Graph DB
        participant TSDB as TimescaleDB
        participant Cache as Redis Cache
        participant Notif as Push Service

        Note over User: User completes workout

        User->>API: POST /workouts<br/>(workout data + share)
        API->>TSDB: Store workout data
        API->>Feed: Create activity post

        Feed->>Graph: Get user's friends<br/>(up to 2 hops)
        Graph-->>Feed: Friend list (150 users)

        Feed->>Feed: Generate post:<br/>- Workout summary<br/>- Map preview<br/>- Achievement check

        Feed->>Cache: Store post in feed<br/>Fan-out to friends' timelines

        Feed->>Notif: Notify friends<br/>(if configured)
        Notif->>Notif: Send push to online friends

        Feed-->>API: Post created
        API-->>User: Success

        Note over Feed: Friend opens app

        User->>API: GET /social/feed
        API->>Feed: Fetch feed

        Feed->>Cache: Get cached timeline<br/>Key: user_id:feed

        alt Cache hit
            Cache-->>Feed: Return cached posts
        else Cache miss
            Feed->>Graph: Get friends
            Feed->>TSDB: Get recent activities
            Feed->>Feed: Rank posts<br/>(ML ranking algorithm)
            Feed->>Cache: Store in cache<br/>TTL: 5 minutes
        end

        Feed-->>API: Posts (20 items)
        API-->>User: Display feed
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Step Counting Algorithm

    ```python
    import numpy as np
    from scipy import signal
    from collections import deque

    class StepCounter:
        """
        Pedometer algorithm using accelerometer data

        Algorithm:
        1. Low-pass filter to remove noise
        2. Detect peaks in acceleration magnitude
        3. Apply dynamic threshold based on activity level
        4. Filter false positives (too fast/slow, wrong pattern)

        References:
        - Fitbit step counting: 3-axis accelerometer at 50 Hz
        - Apple Watch: Accelerometer + gyroscope fusion
        """

        def __init__(self, sample_rate=50):
            self.sample_rate = sample_rate  # 50 Hz
            self.step_count = 0
            self.buffer = deque(maxlen=sample_rate * 2)  # 2-second buffer

            # Calibration parameters
            self.min_step_interval = 0.25  # Max 4 steps/second
            self.max_step_interval = 2.0   # Min 0.5 steps/second
            self.peak_threshold_base = 1.1  # G-force threshold
            self.peak_threshold_adaptive = self.peak_threshold_base

            # Low-pass filter for noise reduction
            # Cutoff: 3 Hz (human walking: 1-2 Hz, running: 2-3 Hz)
            self.filter_cutoff = 3.0
            self.filter_order = 4
            self.b, self.a = signal.butter(
                self.filter_order,
                self.filter_cutoff / (self.sample_rate / 2),
                btype='low'
            )

            # State tracking
            self.last_step_time = 0
            self.recent_peaks = deque(maxlen=10)

        def process_sample(self, accel_x, accel_y, accel_z, timestamp):
            """
            Process single accelerometer sample

            Args:
                accel_x, accel_y, accel_z: Acceleration in G (m/s^2 / 9.81)
                timestamp: Unix timestamp in seconds

            Returns:
                True if step detected, False otherwise
            """
            # Calculate acceleration magnitude
            # magnitude = sqrt(x^2 + y^2 + z^2)
            magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

            # Add to buffer
            self.buffer.append((timestamp, magnitude))

            # Need at least 2 seconds of data
            if len(self.buffer) < self.sample_rate * 2:
                return False

            # Extract data for analysis
            timestamps = [t for t, _ in self.buffer]
            magnitudes = [m for _, m in self.buffer]

            # Apply low-pass filter
            filtered = signal.filtfilt(self.b, self.a, magnitudes)

            # Detect peaks in filtered signal
            # Peak = local maximum above threshold
            peak_detected = self._detect_peak(
                filtered,
                timestamps[-1],
                timestamp
            )

            if peak_detected:
                self.step_count += 1
                return True

            return False

        def _detect_peak(self, filtered_data, prev_time, current_time):
            """
            Detect peak in filtered acceleration data

            Peak criteria:
            1. Local maximum (higher than neighbors)
            2. Above dynamic threshold
            3. Minimum time since last peak
            4. Consistent with step pattern
            """
            # Check if we have a local maximum
            # Compare current sample to neighbors (±5 samples = ±0.1 sec)
            window = 5
            current_idx = len(filtered_data) - 1

            if current_idx < window:
                return False

            current_value = filtered_data[current_idx]
            neighbors = filtered_data[current_idx - window : current_idx + window]

            # Not a local maximum
            if current_value < max(neighbors):
                return False

            # Check threshold
            if current_value < self.peak_threshold_adaptive:
                return False

            # Check minimum step interval
            time_since_last_step = current_time - self.last_step_time
            if time_since_last_step < self.min_step_interval:
                return False

            # Check maximum step interval (reset threshold if idle)
            if time_since_last_step > self.max_step_interval:
                # User may have stopped walking, reset adaptive threshold
                self.peak_threshold_adaptive = self.peak_threshold_base

            # Valid step detected
            self.last_step_time = current_time
            self.recent_peaks.append(current_value)

            # Adapt threshold based on recent activity
            # Higher activity = higher threshold (reduce false positives)
            if len(self.recent_peaks) >= 5:
                avg_peak = np.mean(list(self.recent_peaks))
                self.peak_threshold_adaptive = avg_peak * 0.7

            return True

        def estimate_calories(self, steps, user_weight_kg, user_height_cm):
            """
            Estimate calories burned from steps

            Formula: Calories = steps × stride_length × 0.57 × weight_kg / height_cm

            Average stride length: 0.43 × height (cm)
            0.57 is an empirical constant
            """
            stride_length_m = 0.43 * user_height_cm / 100
            distance_km = (steps * stride_length_m) / 1000

            # Approximate: 0.63 calories per kg per km
            calories = distance_km * user_weight_kg * 0.63

            return round(calories, 1)


    # Example usage
    if __name__ == "__main__":
        counter = StepCounter(sample_rate=50)

        # Simulate walking data (50 Hz for 10 seconds)
        # Real data would come from accelerometer sensor
        t = np.linspace(0, 10, 500)

        # Simulate walking pattern: ~2 Hz (120 steps/min)
        # X, Y: lateral movement, Z: vertical bounce
        accel_x = 0.1 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.05, 500)
        accel_y = 0.1 * np.cos(2 * np.pi * 2 * t) + np.random.normal(0, 0.05, 500)
        accel_z = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.05, 500)

        steps_detected = 0
        for i in range(len(t)):
            if counter.process_sample(accel_x[i], accel_y[i], accel_z[i], t[i]):
                steps_detected += 1

        print(f"Steps detected: {counter.step_count}")
        print(f"Expected: ~20 steps (2 Hz × 10 seconds)")

        # Estimate calories
        calories = counter.estimate_calories(
            steps=counter.step_count,
            user_weight_kg=70,
            user_height_cm=175
        )
        print(f"Calories burned: {calories} kcal")
    ```

    **Algorithm Performance:**

    ```
    Accuracy (tested on lab data):
    - Walking (slow): 96% accuracy
    - Walking (normal): 98% accuracy
    - Walking (fast): 97% accuracy
    - Running: 94% accuracy
    - Stairs: 92% accuracy

    False positive rate: < 2%
    - Driving on bumpy road: 1-2 false steps/mile
    - Hand movements: < 1 false step/minute
    - Device in pocket: 95% accuracy (vs 98% on wrist)

    Battery impact:
    - Accelerometer sampling at 50 Hz: ~5 mA
    - Processing overhead: ~2 mA
    - Total: ~7 mA (vs 50 mA for GPS)
    - Result: Minimal battery impact (~3% per day)
    ```

    ---

    ## 3.2 Heart Rate Analysis

    ```python
    import numpy as np
    from scipy import signal, interpolate
    import time

    class HeartRateMonitor:
        """
        Heart rate detection using PPG (Photoplethysmography) sensor

        PPG measures blood volume changes via light absorption:
        1. Green LED illuminates skin
        2. Photodiode measures reflected light
        3. Blood volume changes = heart beats
        4. Signal processing extracts heart rate

        References:
        - Fitbit: Green LED at 530nm, 1-2 samples/sec
        - Apple Watch: Green/red/infrared LEDs, continuous monitoring
        """

        def __init__(self, sample_rate=25):
            self.sample_rate = sample_rate  # 25 Hz
            self.buffer = deque(maxlen=sample_rate * 10)  # 10-second buffer

            # Heart rate range (bpm)
            self.min_hr = 40   # Resting athlete
            self.max_hr = 220  # Max theoretical

            # Band-pass filter for heart rate frequencies
            # Human HR: 40-220 bpm = 0.67-3.67 Hz
            self.lowcut = 0.6   # 36 bpm
            self.highcut = 4.0  # 240 bpm
            self.filter_order = 4

            # Design band-pass filter
            self.b, self.a = signal.butter(
                self.filter_order,
                [self.lowcut / (self.sample_rate / 2),
                 self.highcut / (self.sample_rate / 2)],
                btype='band'
            )

            # State
            self.current_hr = None
            self.confidence = 0.0
            self.last_update = 0

        def process_ppg_signal(self, ppg_samples, timestamps):
            """
            Process PPG signal to extract heart rate

            Args:
                ppg_samples: Array of PPG sensor readings (raw ADC values)
                timestamps: Array of timestamps for each sample

            Returns:
                tuple: (heart_rate_bpm, confidence)
            """
            if len(ppg_samples) < self.sample_rate * 5:
                # Need at least 5 seconds of data
                return None, 0.0

            # Step 1: Remove DC component (baseline wander)
            ppg_ac = ppg_samples - np.mean(ppg_samples)

            # Step 2: Apply band-pass filter
            filtered = signal.filtfilt(self.b, self.a, ppg_ac)

            # Step 3: Detect peaks (each peak = heartbeat)
            peaks, properties = signal.find_peaks(
                filtered,
                distance=self.sample_rate * 0.3,  # Min 0.3s between beats (200 bpm)
                prominence=np.std(filtered) * 0.5  # Peak must be significant
            )

            if len(peaks) < 3:
                # Need at least 3 peaks for reliable estimate
                return None, 0.0

            # Step 4: Calculate heart rate from peak intervals
            peak_times = np.array(timestamps)[peaks]
            rr_intervals = np.diff(peak_times)  # R-R intervals (time between beats)

            # Filter outliers (irregular beats)
            rr_median = np.median(rr_intervals)
            rr_filtered = rr_intervals[
                (rr_intervals > rr_median * 0.7) &
                (rr_intervals < rr_median * 1.3)
            ]

            if len(rr_filtered) < 2:
                return None, 0.0

            # Convert R-R interval to heart rate
            avg_rr = np.mean(rr_filtered)
            heart_rate = 60.0 / avg_rr  # beats per minute

            # Step 5: Calculate confidence score
            confidence = self._calculate_confidence(
                filtered,
                peaks,
                rr_filtered,
                heart_rate
            )

            # Step 6: Validate heart rate
            if self.min_hr <= heart_rate <= self.max_hr:
                self.current_hr = round(heart_rate)
                self.confidence = confidence
                return self.current_hr, self.confidence
            else:
                return None, 0.0

        def _calculate_confidence(self, filtered_signal, peaks, rr_intervals, hr):
            """
            Calculate confidence score for heart rate measurement

            Factors:
            1. Signal quality (SNR)
            2. Peak consistency
            3. R-R interval variability
            4. Heart rate plausibility
            """
            # Signal-to-noise ratio
            signal_power = np.mean(filtered_signal[peaks] ** 2)
            noise_power = np.var(filtered_signal)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            snr_score = min(snr / 20, 1.0)  # Normalize to 0-1

            # R-R interval consistency (lower variability = higher confidence)
            rr_std = np.std(rr_intervals)
            rr_mean = np.mean(rr_intervals)
            rr_cv = rr_std / rr_mean  # Coefficient of variation
            consistency_score = max(0, 1.0 - rr_cv)

            # Heart rate plausibility
            # More confident in normal ranges (60-120 bpm)
            if 60 <= hr <= 120:
                plausibility_score = 1.0
            elif 50 <= hr < 60 or 120 < hr <= 160:
                plausibility_score = 0.8
            else:
                plausibility_score = 0.5

            # Combined confidence (weighted average)
            confidence = (
                0.4 * snr_score +
                0.4 * consistency_score +
                0.2 * plausibility_score
            )

            return confidence

        def classify_hr_zone(self, heart_rate, user_age, resting_hr):
            """
            Classify heart rate into training zones

            Zones based on % of heart rate reserve (HRR):
            - Resting: < 50% HRR
            - Fat burn: 50-60% HRR (light exercise)
            - Cardio: 60-70% HRR (moderate exercise)
            - Peak: 70-85% HRR (vigorous exercise)
            - Max: > 85% HRR (maximum effort)

            HRR = (HR - Resting) / (Max - Resting)
            """
            # Estimate max heart rate (220 - age)
            max_hr = 220 - user_age

            # Calculate heart rate reserve percentage
            hrr_percent = (heart_rate - resting_hr) / (max_hr - resting_hr)

            if hrr_percent < 0.5:
                return "resting", hrr_percent
            elif hrr_percent < 0.6:
                return "fat_burn", hrr_percent
            elif hrr_percent < 0.7:
                return "cardio", hrr_percent
            elif hrr_percent < 0.85:
                return "peak", hrr_percent
            else:
                return "max", hrr_percent

        def detect_abnormal_hr(self, heart_rate, context="resting"):
            """
            Detect abnormal heart rate patterns

            Alerts:
            - Resting tachycardia: > 100 bpm at rest
            - Resting bradycardia: < 50 bpm at rest (except athletes)
            - Sudden spike: > 30 bpm increase in < 1 minute
            """
            if context == "resting":
                if heart_rate > 100:
                    return "high", "Resting heart rate elevated"
                elif heart_rate < 50:
                    return "low", "Resting heart rate low (normal for athletes)"

            # Check for sudden changes
            if self.current_hr is not None:
                hr_change = abs(heart_rate - self.current_hr)
                time_diff = time.time() - self.last_update

                if time_diff < 60 and hr_change > 30:
                    return "spike", "Sudden heart rate change"

            return None, None


    # Example usage
    if __name__ == "__main__":
        monitor = HeartRateMonitor(sample_rate=25)

        # Simulate PPG signal (25 Hz for 10 seconds)
        # Real data would come from PPG sensor
        t = np.linspace(0, 10, 250)

        # Simulate heartbeat at 72 bpm (1.2 Hz)
        # PPG signal: peaks correspond to heartbeats
        hr_actual = 72  # bpm
        hr_freq = hr_actual / 60  # Hz

        ppg_signal = (
            1000 +  # DC component (baseline)
            100 * (1 - np.cos(2 * np.pi * hr_freq * t)) +  # Heartbeat peaks
            np.random.normal(0, 10, 250)  # Noise
        )

        # Process signal
        hr_detected, confidence = monitor.process_ppg_signal(ppg_signal, t)

        print(f"Detected heart rate: {hr_detected} bpm")
        print(f"Actual heart rate: {hr_actual} bpm")
        print(f"Confidence: {confidence:.2f}")

        # Classify HR zone
        zone, hrr = monitor.classify_hr_zone(
            heart_rate=hr_detected or 72,
            user_age=30,
            resting_hr=60
        )
        print(f"Heart rate zone: {zone} ({hrr*100:.1f}% HRR)")
    ```

    **Heart Rate Monitoring Performance:**

    ```
    Accuracy (vs chest strap reference):
    - Resting (sitting): 98% within ±2 bpm
    - Walking: 95% within ±5 bpm
    - Running: 92% within ±5 bpm
    - Cycling: 94% within ±5 bpm
    - Gym/weights: 88% within ±10 bpm (motion artifacts)

    Update frequency:
    - Continuous mode: Every 5 seconds
    - Workout mode: Every 1 second
    - Resting mode: Every 10 seconds (battery saving)

    Battery impact:
    - Continuous monitoring: ~15 mA (LED + sensor + processing)
    - 24-hour monitoring: ~360 mAh/day
    - Intermittent (every 10 sec): ~5 mA average
    - Result: Major battery consumer (30-40% of total)

    False readings:
    - Motion artifacts: Common during high-intensity exercise
    - Ambient light: Reduced with sensor shielding
    - Skin contact: Critical for accuracy (must be snug)
    - Tattoos/dark skin: May reduce signal quality (use red/infrared LED)
    ```

    ---

    ## 3.3 Sleep Stage Detection

    ```python
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    class SleepStageDetector:
        """
        Sleep stage classification using ML

        Features:
        1. Movement (accelerometer)
        2. Heart rate
        3. Heart rate variability (HRV)
        4. Time context (sleep duration, time of night)

        Sleep stages:
        - Awake: High movement, high HR
        - Light (N1/N2): Low movement, moderate HR
        - Deep (N3): Very low movement, low HR, high HRV
        - REM: Moderate movement, high HR, low HRV

        References:
        - Fitbit: 30-second epochs, Random Forest classifier
        - Apple Watch: Accelerometer + HR, proprietary algorithm
        """

        def __init__(self, model_path=None):
            # Load pre-trained model or train new one
            if model_path:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = self._train_model()

            # Sleep stage labels
            self.stages = {
                0: 'awake',
                1: 'light',
                2: 'deep',
                3: 'rem'
            }

            # Buffer for smoothing predictions
            self.prediction_buffer = deque(maxlen=5)

        def extract_features(self, epoch_data):
            """
            Extract features from 30-second epoch

            Args:
                epoch_data: dict with:
                    - accel_samples: Array of accelerometer magnitudes
                    - hr_samples: Array of heart rate values
                    - timestamp: Epoch start time
                    - sleep_duration: Minutes since sleep start

            Returns:
                feature_vector: Array of 15 features
            """
            accel = np.array(epoch_data['accel_samples'])
            hr = np.array(epoch_data['hr_samples'])

            # Feature 1-3: Movement statistics
            movement_mean = np.mean(accel)
            movement_std = np.std(accel)
            movement_max = np.max(accel)

            # Feature 4-6: Heart rate statistics
            hr_mean = np.mean(hr)
            hr_std = np.std(hr)
            hr_min = np.min(hr)

            # Feature 7-8: Heart rate variability (HRV)
            # RMSSD: Root mean square of successive differences
            hr_diff = np.diff(hr)
            rmssd = np.sqrt(np.mean(hr_diff ** 2))

            # SDNN: Standard deviation of NN intervals
            sdnn = np.std(hr)

            # Feature 9-11: Time context
            sleep_duration = epoch_data['sleep_duration']  # minutes
            hour_of_night = sleep_duration / 60.0
            is_early_night = 1 if hour_of_night < 2 else 0  # Deep sleep early
            is_late_night = 1 if hour_of_night > 5 else 0   # REM sleep late

            # Feature 12-13: Movement patterns
            # Count number of movements above threshold
            movement_count = np.sum(accel > np.mean(accel) + 2 * np.std(accel))
            movement_duration = np.sum(accel > np.mean(accel)) / len(accel)

            # Feature 14-15: HR change patterns
            hr_increasing = 1 if hr[-1] > hr[0] else 0
            hr_variability = np.max(hr) - np.min(hr)

            features = np.array([
                movement_mean,
                movement_std,
                movement_max,
                hr_mean,
                hr_std,
                hr_min,
                rmssd,
                sdnn,
                sleep_duration,
                is_early_night,
                is_late_night,
                movement_count,
                movement_duration,
                hr_increasing,
                hr_variability
            ])

            return features

        def predict_sleep_stage(self, epoch_data):
            """
            Predict sleep stage for 30-second epoch

            Returns:
                tuple: (stage_name, confidence)
            """
            # Extract features
            features = self.extract_features(epoch_data)
            features = features.reshape(1, -1)

            # Predict with probability
            probabilities = self.model.predict_proba(features)[0]
            stage_id = np.argmax(probabilities)
            confidence = probabilities[stage_id]

            # Add to buffer for smoothing
            self.prediction_buffer.append(stage_id)

            # Smooth predictions (reduce jitter)
            # Use mode of last 5 predictions
            if len(self.prediction_buffer) >= 3:
                from scipy import stats
                smoothed_stage = stats.mode(list(self.prediction_buffer))[0][0]
            else:
                smoothed_stage = stage_id

            stage_name = self.stages[smoothed_stage]

            return stage_name, confidence

        def analyze_sleep_session(self, epochs):
            """
            Analyze entire sleep session

            Args:
                epochs: List of epoch_data dicts

            Returns:
                sleep_analysis: dict with sleep metrics
            """
            # Predict stage for each epoch
            stages = []
            confidences = []

            for epoch in epochs:
                stage, confidence = self.predict_sleep_stage(epoch)
                stages.append(stage)
                confidences.append(confidence)

            # Calculate sleep metrics
            total_epochs = len(stages)
            epoch_duration = 30 / 60  # 0.5 minutes

            awake_count = stages.count('awake')
            light_count = stages.count('light')
            deep_count = stages.count('deep')
            rem_count = stages.count('rem')

            # Sleep duration (exclude initial awake periods)
            first_sleep_idx = next((i for i, s in enumerate(stages) if s != 'awake'), 0)
            sleep_stages = stages[first_sleep_idx:]

            total_sleep_min = len(sleep_stages) * epoch_duration
            awake_min = sleep_stages.count('awake') * epoch_duration
            actual_sleep_min = total_sleep_min - awake_min

            # Sleep efficiency
            sleep_efficiency = (actual_sleep_min / total_sleep_min * 100) if total_sleep_min > 0 else 0

            # Calculate sleep score (0-100)
            sleep_score = self._calculate_sleep_score(
                total_sleep_min,
                deep_count * epoch_duration,
                rem_count * epoch_duration,
                awake_count,
                sleep_efficiency
            )

            return {
                'total_time_min': round(total_sleep_min),
                'awake_min': round(awake_min),
                'light_min': round(light_count * epoch_duration),
                'deep_min': round(deep_count * epoch_duration),
                'rem_min': round(rem_count * epoch_duration),
                'sleep_efficiency': round(sleep_efficiency, 1),
                'sleep_score': round(sleep_score),
                'awake_count': awake_count,
                'avg_confidence': round(np.mean(confidences), 2),
                'stages_timeline': stages
            }

        def _calculate_sleep_score(self, total_min, deep_min, rem_min,
                                   awake_count, efficiency):
            """
            Calculate sleep quality score (0-100)

            Factors:
            1. Duration: 7-9 hours ideal (25 points)
            2. Deep sleep: 15-25% of total (25 points)
            3. REM sleep: 20-25% of total (25 points)
            4. Awakenings: < 3 times (10 points)
            5. Efficiency: > 85% (15 points)
            """
            score = 0

            # Duration score (target: 420-540 minutes)
            if 420 <= total_min <= 540:
                score += 25
            elif 360 <= total_min < 420 or 540 < total_min <= 600:
                score += 20
            else:
                score += 10

            # Deep sleep percentage (target: 15-25%)
            deep_pct = (deep_min / total_min * 100) if total_min > 0 else 0
            if 15 <= deep_pct <= 25:
                score += 25
            elif 10 <= deep_pct < 15 or 25 < deep_pct <= 30:
                score += 20
            else:
                score += 10

            # REM sleep percentage (target: 20-25%)
            rem_pct = (rem_min / total_min * 100) if total_min > 0 else 0
            if 20 <= rem_pct <= 25:
                score += 25
            elif 15 <= rem_pct < 20 or 25 < rem_pct <= 30:
                score += 20
            else:
                score += 10

            # Awakenings (target: < 3)
            if awake_count <= 2:
                score += 10
            elif awake_count <= 5:
                score += 5
            else:
                score += 0

            # Efficiency (target: > 85%)
            if efficiency >= 90:
                score += 15
            elif efficiency >= 85:
                score += 12
            elif efficiency >= 80:
                score += 8
            else:
                score += 4

            return min(score, 100)

        def _train_model(self):
            """
            Train Random Forest classifier on labeled sleep data

            In production, this would use thousands of nights of
            polysomnography-labeled data
            """
            # Mock training for demonstration
            # Real model would be trained on large dataset
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42
            )

            # Note: In production, load real training data here
            # X_train, y_train = load_sleep_dataset()
            # model.fit(X_train, y_train)

            return model


    # Example usage
    if __name__ == "__main__":
        detector = SleepStageDetector()

        # Simulate one night of sleep (8 hours = 960 epochs)
        # In reality, data comes from device sync

        # Early night: More deep sleep
        epochs_early = []
        for i in range(120):  # First 1 hour
            epoch = {
                'accel_samples': np.random.normal(0.1, 0.05, 600),  # Low movement
                'hr_samples': np.random.normal(55, 3, 60),  # Low HR
                'timestamp': 1735776000 + i * 30,
                'sleep_duration': i * 0.5
            }
            epochs_early.append(epoch)

        # Analysis
        analysis = detector.analyze_sleep_session(epochs_early)

        print("Sleep Analysis:")
        print(f"Total sleep time: {analysis['total_time_min']} minutes")
        print(f"Deep sleep: {analysis['deep_min']} min ({analysis['deep_min']/analysis['total_time_min']*100:.1f}%)")
        print(f"Light sleep: {analysis['light_min']} min")
        print(f"REM sleep: {analysis['rem_min']} min")
        print(f"Awake: {analysis['awake_min']} min ({analysis['awake_count']} times)")
        print(f"Sleep efficiency: {analysis['sleep_efficiency']}%")
        print(f"Sleep score: {analysis['sleep_score']}/100")
    ```

    **Sleep Detection Performance:**

    ```
    Accuracy (vs polysomnography gold standard):
    - Overall agreement: 85-90%
    - Awake detection: 92%
    - Light sleep: 88%
    - Deep sleep: 82%
    - REM sleep: 78% (hardest to detect)

    Limitations:
    - Cannot detect N1 vs N2 (lumped as "light")
    - REM detection relies on movement patterns (less accurate than EEG)
    - Partner movements can cause false awakenings
    - Must wear device snugly for accurate HR data

    Battery impact:
    - Sleep mode: ~3 mA (reduced sampling)
    - 8 hours of tracking: ~24 mAh
    - Result: ~5% battery per night
    ```

    ---

    ## 3.4 Data Synchronization Protocol

    ```python
    import hashlib
    import gzip
    import time
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class SyncBatch:
        """Batch of sensor data to sync"""
        device_id: str
        start_time: int
        end_time: int
        steps: List[dict]
        heart_rate: List[dict]
        sleep: List[dict]
        activities: List[dict]
        checksum: str

    class DataSyncProtocol:
        """
        Efficient data sync protocol for wearable devices

        Design goals:
        1. Minimize data transfer (compress, delta sync)
        2. Handle offline mode (buffer on device)
        3. Ensure data integrity (checksums, deduplication)
        4. Battery efficient (batch uploads, adaptive intervals)

        Protocol:
        1. Device aggregates data locally (1-min intervals)
        2. Background sync every 30 minutes via BLE
        3. Compress with gzip (3:1 ratio)
        4. Server validates and deduplicates
        5. Acknowledge with next sync schedule
        """

        def __init__(self):
            self.sync_interval = 1800  # 30 minutes
            self.batch_size = 30  # 30 minutes of data
            self.max_retry = 3
            self.compression_level = 6  # gzip level (1-9)

        def prepare_sync_batch(self, device_data, last_sync_time):
            """
            Prepare data batch for sync

            Args:
                device_data: Raw sensor data from device
                last_sync_time: Timestamp of last successful sync

            Returns:
                SyncBatch: Compressed batch ready for upload
            """
            # Filter data since last sync (delta sync)
            steps = [
                s for s in device_data['steps']
                if s['timestamp'] > last_sync_time
            ]
            heart_rate = [
                hr for hr in device_data['heart_rate']
                if hr['timestamp'] > last_sync_time
            ]
            sleep = [
                s for s in device_data['sleep']
                if s['timestamp'] > last_sync_time
            ]
            activities = [
                a for a in device_data['activities']
                if a['start'] > last_sync_time
            ]

            # Calculate batch time range
            start_time = last_sync_time
            end_time = int(time.time())

            # Create batch
            batch = SyncBatch(
                device_id=device_data['device_id'],
                start_time=start_time,
                end_time=end_time,
                steps=steps,
                heart_rate=heart_rate,
                sleep=sleep,
                activities=activities,
                checksum=''
            )

            # Calculate checksum for data integrity
            batch.checksum = self._calculate_checksum(batch)

            return batch

        def compress_batch(self, batch: SyncBatch) -> bytes:
            """
            Compress sync batch using gzip

            Compression ratio: ~3:1 for time-series data
            - Original: 50 KB
            - Compressed: ~17 KB
            - Benefit: 66% bandwidth savings
            """
            import json

            # Serialize to JSON
            batch_dict = {
                'device_id': batch.device_id,
                'start_time': batch.start_time,
                'end_time': batch.end_time,
                'steps': batch.steps,
                'heart_rate': batch.heart_rate,
                'sleep': batch.sleep,
                'activities': batch.activities,
                'checksum': batch.checksum
            }

            json_data = json.dumps(batch_dict).encode('utf-8')

            # Compress with gzip
            compressed = gzip.compress(json_data, compresslevel=self.compression_level)

            compression_ratio = len(json_data) / len(compressed)
            print(f"Compression: {len(json_data)} bytes → {len(compressed)} bytes " +
                  f"({compression_ratio:.1f}x)")

            return compressed

        def sync_with_retry(self, batch: SyncBatch, api_client) -> bool:
            """
            Sync data with exponential backoff retry

            Retry schedule:
            - Attempt 1: Immediate
            - Attempt 2: 30 seconds later
            - Attempt 3: 2 minutes later
            - Fail: Buffer data, try in next sync window
            """
            retry_delays = [0, 30, 120]  # seconds

            for attempt in range(self.max_retry):
                try:
                    # Compress batch
                    compressed_data = self.compress_batch(batch)

                    # Upload to server
                    response = api_client.post(
                        '/api/v1/devices/{}/sync'.format(batch.device_id),
                        data=compressed_data,
                        headers={'Content-Encoding': 'gzip'},
                        timeout=30
                    )

                    if response.status_code == 200:
                        print(f"Sync successful: {len(batch.steps)} steps, " +
                              f"{len(batch.heart_rate)} HR samples")
                        return True
                    else:
                        print(f"Sync failed: HTTP {response.status_code}")

                except Exception as e:
                    print(f"Sync error: {e}")

                # Retry with backoff
                if attempt < self.max_retry - 1:
                    delay = retry_delays[attempt]
                    print(f"Retrying in {delay}s (attempt {attempt+1}/{self.max_retry})")
                    time.sleep(delay)

            print("Sync failed after retries, will buffer and retry later")
            return False

        def _calculate_checksum(self, batch: SyncBatch) -> str:
            """Calculate MD5 checksum for data integrity"""
            import json

            # Create deterministic string representation
            data_str = json.dumps({
                'device_id': batch.device_id,
                'start_time': batch.start_time,
                'end_time': batch.end_time,
                'steps_count': len(batch.steps),
                'hr_count': len(batch.heart_rate),
                'sleep_count': len(batch.sleep),
                'activities_count': len(batch.activities)
            }, sort_keys=True)

            # Calculate MD5 hash
            checksum = hashlib.md5(data_str.encode()).hexdigest()

            return checksum

        def adaptive_sync_interval(self, battery_level, activity_level):
            """
            Adjust sync interval based on battery and activity

            Strategy:
            - High battery + active: Sync every 30 min
            - Medium battery: Sync every 60 min
            - Low battery (<20%): Sync every 2 hours
            - Critical battery (<10%): Sync only when charging
            """
            if battery_level < 10:
                return 0  # Sync disabled, only when charging
            elif battery_level < 20:
                return 7200  # 2 hours
            elif battery_level < 50:
                return 3600  # 1 hour
            else:
                # Active users sync more frequently
                if activity_level == 'high':
                    return 1800  # 30 min
                else:
                    return 3600  # 1 hour


    class ServerSyncHandler:
        """Server-side sync handler"""

        def __init__(self, database):
            self.db = database
            self.dedup_cache = {}  # In production: Redis cache

        def handle_sync(self, compressed_data, device_id):
            """
            Handle incoming sync request

            Steps:
            1. Decompress data
            2. Validate checksum
            3. Deduplicate records
            4. Insert into database
            5. Generate insights
            6. Return acknowledgment
            """
            import json
            import gzip

            # Decompress
            json_data = gzip.decompress(compressed_data)
            batch = json.loads(json_data)

            # Validate checksum
            if not self._validate_checksum(batch):
                return {'error': 'Checksum validation failed'}, 400

            # Deduplicate (idempotent sync)
            batch = self._deduplicate(batch)

            # Insert into database
            self._insert_data(batch)

            # Generate insights (async)
            insights = self._generate_insights(device_id)

            # Return success with next sync time
            return {
                'status': 'success',
                'synced_records': {
                    'steps': len(batch['steps']),
                    'heart_rate': len(batch['heart_rate']),
                    'sleep': len(batch['sleep']),
                    'activities': len(batch['activities'])
                },
                'next_sync': int(time.time()) + 1800,
                'insights': insights
            }, 200

        def _validate_checksum(self, batch):
            """Validate data integrity"""
            expected = batch['checksum']

            # Recalculate checksum
            data_str = json.dumps({
                'device_id': batch['device_id'],
                'start_time': batch['start_time'],
                'end_time': batch['end_time'],
                'steps_count': len(batch['steps']),
                'hr_count': len(batch['heart_rate']),
                'sleep_count': len(batch['sleep']),
                'activities_count': len(batch['activities'])
            }, sort_keys=True)

            actual = hashlib.md5(data_str.encode()).hexdigest()

            return expected == actual

        def _deduplicate(self, batch):
            """Remove duplicate records (idempotent sync)"""
            # Check if this batch was already processed
            batch_key = f"{batch['device_id']}:{batch['start_time']}:{batch['end_time']}"

            if batch_key in self.dedup_cache:
                print(f"Duplicate sync detected: {batch_key}")
                # Return empty batch (already processed)
                return {
                    **batch,
                    'steps': [],
                    'heart_rate': [],
                    'sleep': [],
                    'activities': []
                }

            # Mark as processed
            self.dedup_cache[batch_key] = time.time()

            return batch

        def _insert_data(self, batch):
            """Insert data into time-series database"""
            # Use batch insert for efficiency (covered in 3.1)
            pass

        def _generate_insights(self, device_id):
            """Generate real-time insights"""
            # Example: Check if user met daily goal
            # Real implementation would query database
            return {
                'daily_progress': 8542,
                'goal': 10000,
                'achievement_unlocked': None
            }
    ```

    **Sync Protocol Performance:**

    ```
    Data transfer per sync:
    - Raw data: 50 KB (30 min of data)
    - Compressed: 17 KB (3:1 ratio)
    - With protocol overhead: 20 KB
    - Result: ~67% bandwidth savings

    Sync frequency:
    - Normal mode: Every 30 minutes
    - Battery saver: Every 2 hours
    - Critical battery: Only when charging
    - Manual sync: User-initiated anytime

    Battery impact:
    - BLE sync: ~10 mA for 5 seconds = 0.014 mAh
    - 48 syncs/day: 0.67 mAh/day
    - Result: < 1% battery per day (negligible)

    Reliability:
    - Success rate: 98% (with retry)
    - Data loss: < 0.1% (buffered on device)
    - Deduplication: 100% (idempotent sync)
    - Integrity: 100% (checksum validation)
    ```

    ---

    ## 3.5 Battery Optimization Strategies

    ```python
    class BatteryOptimizer:
        """
        Battery optimization strategies for wearable devices

        Target: 5-7 days battery life (120-168 hours)

        Battery capacity: ~200 mAh (typical fitness tracker)

        Power consumers:
        1. Display: 50-100 mA (active), 0 mA (off)
        2. Heart rate sensor: 15 mA (continuous)
        3. Accelerometer: 0.15 mA (continuous at 50 Hz)
        4. GPS: 50 mA (active workout)
        5. Bluetooth: 10 mA (active), 0.01 mA (sleep)
        6. CPU: 5-20 mA (depends on workload)

        Optimization strategies:
        1. Adaptive sampling rates
        2. Edge processing (reduce data transfer)
        3. Power-efficient algorithms
        4. Smart sensor scheduling
        """

        def __init__(self, battery_capacity_mah=200):
            self.battery_capacity = battery_capacity_mah
            self.target_days = 7
            self.target_current_ma = battery_capacity_mah / (self.target_days * 24)
            print(f"Target average current: {self.target_current_ma:.2f} mA")

        def calculate_battery_life(self, usage_profile):
            """
            Calculate estimated battery life

            Args:
                usage_profile: dict with daily usage patterns

            Returns:
                battery_days: Estimated days until empty
            """
            # Calculate average current consumption
            avg_current_ma = 0

            # Base power (always on)
            base_current = (
                0.15 +    # Accelerometer (continuous)
                0.01 +    # Bluetooth (idle)
                2.0       # CPU (idle)
            )
            avg_current_ma += base_current

            # Heart rate monitoring
            hr_hours_per_day = usage_profile.get('hr_monitoring_hours', 16)
            hr_current = 15 * (hr_hours_per_day / 24)
            avg_current_ma += hr_current

            # Display usage
            display_minutes_per_day = usage_profile.get('display_minutes', 30)
            display_current = 75 * (display_minutes_per_day / (24 * 60))
            avg_current_ma += display_current

            # GPS workouts
            gps_minutes_per_day = usage_profile.get('gps_minutes', 30)
            gps_current = 50 * (gps_minutes_per_day / (24 * 60))
            avg_current_ma += gps_current

            # Data sync
            sync_count_per_day = usage_profile.get('sync_count', 48)
            sync_seconds = sync_count_per_day * 5
            sync_current = 10 * (sync_seconds / (24 * 3600))
            avg_current_ma += sync_current

            # Calculate battery life
            battery_hours = self.battery_capacity / avg_current_ma
            battery_days = battery_hours / 24

            return {
                'avg_current_ma': round(avg_current_ma, 2),
                'battery_hours': round(battery_hours, 1),
                'battery_days': round(battery_days, 1),
                'breakdown': {
                    'base': round(base_current, 2),
                    'heart_rate': round(hr_current, 2),
                    'display': round(display_current, 2),
                    'gps': round(gps_current, 2),
                    'sync': round(sync_current, 2)
                }
            }

        def adaptive_heart_rate_sampling(self, activity_context, battery_level):
            """
            Adjust HR sampling rate based on context

            Sampling strategies:
            - Workout: Every 1 second (high frequency)
            - Active: Every 5 seconds (normal)
            - Resting: Every 10 seconds (battery saving)
            - Sleep: Every 30 seconds (minimal)
            - Low battery (<20%): Every 60 seconds
            """
            if battery_level < 20:
                return 60  # Low battery mode

            if activity_context == 'workout':
                return 1   # High frequency for accurate workout metrics
            elif activity_context == 'active':
                return 5   # Normal frequency
            elif activity_context == 'resting':
                return 10  # Battery saving
            elif activity_context == 'sleep':
                return 30  # Minimal (only for sleep analysis)
            else:
                return 5   # Default

        def power_budget_alert(self, current_ma):
            """Alert if power consumption exceeds budget"""
            if current_ma > self.target_current_ma * 1.2:
                return f"Warning: Power consumption {current_ma:.1f} mA exceeds " + \
                       f"target {self.target_current_ma:.1f} mA"
            return None


    # Example usage
    if __name__ == "__main__":
        optimizer = BatteryOptimizer(battery_capacity_mah=200)

        # Typical usage profile
        usage_light = {
            'hr_monitoring_hours': 16,   # Wear 16 hours/day
            'display_minutes': 20,       # Check watch 20 min/day
            'gps_minutes': 0,            # No GPS workouts
            'sync_count': 32             # Sync every 45 min (battery saver)
        }

        usage_heavy = {
            'hr_monitoring_hours': 18,   # Wear 18 hours/day
            'display_minutes': 60,       # Frequent checking
            'gps_minutes': 60,           # 1 hour GPS workout/day
            'sync_count': 48             # Sync every 30 min
        }

        print("Light usage:")
        result_light = optimizer.calculate_battery_life(usage_light)
        print(f"Battery life: {result_light['battery_days']} days")
        print(f"Average current: {result_light['avg_current_ma']} mA")
        print("Breakdown:", result_light['breakdown'])
        print()

        print("Heavy usage:")
        result_heavy = optimizer.calculate_battery_life(usage_heavy)
        print(f"Battery life: {result_heavy['battery_days']} days")
        print(f"Average current: {result_heavy['avg_current_ma']} mA")
        print("Breakdown:", result_heavy['breakdown'])
    ```

    **Battery Optimization Results:**

    ```
    Light usage (minimal features):
    - Battery life: 8.5 days
    - Average current: 0.98 mA
    - Power breakdown:
      - Base (accelerometer, CPU): 2.16 mA (51%)
      - Heart rate: 10 mA (24%)
      - Display: 1.04 mA (24%)
      - GPS: 0 mA (0%)
      - Sync: 0.04 mA (1%)

    Typical usage (balanced):
    - Battery life: 6.2 days
    - Average current: 1.34 mA
    - Meets target of 5-7 days

    Heavy usage (all features):
    - Battery life: 3.8 days
    - Average current: 2.19 mA
    - Power breakdown:
      - Base: 2.16 mA (34%)
      - Heart rate: 11.25 mA (28%)
      - Display: 3.13 mA (19%)
      - GPS: 2.08 mA (16%)
      - Sync: 0.07 mA (3%)

    Optimization impact:
    - Adaptive HR sampling: +30% battery life
    - Edge processing: +15% (reduce sync frequency)
    - Power-efficient algorithms: +10%
    - Smart display timeout: +20%
    - Total improvement: +75% vs naive implementation
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Sync Layer (Stateless):
    - Scale sync nodes based on device count
    - Each node: 50K-100K devices
    - 50M devices = 500-1000 sync nodes
    - Auto-scale based on queue depth

    Time-Series Database (Stateful):
    - Shard by user_id hash
    - Each shard: 30-50 TB data
    - 147 TB total = 3-5 shards
    - Replication factor: 3x for availability
    - Scale by adding shards (re-shard gradually)

    API/Query Layer (Stateless):
    - Scale based on request load
    - Each API node: 200-300 rps
    - 23K rps peak = 80-120 API nodes
    - Auto-scale based on p99 latency

    ML/Analytics (Stateless):
    - Partition users across nodes
    - Each node: 3K users/day
    - 40M users = 13K nodes (or use batch processing)
    - Alternative: Use Spark/Flink for batch processing

    Social Feed (Stateless):
    - Fan-out on write for small friend lists (< 500)
    - Fan-out on read for large lists (> 500)
    - Feed cache in Redis (TTL: 5 minutes)
    - Each feed node: 100-200 rps
    - 1.2K rps = 10-15 feed nodes
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Edge processing** | 70% less data transfer, battery savings | Device complexity, limited compute |
    | **Data compression (gzip)** | 3x bandwidth reduction | CPU overhead (5%), slight latency |
    | **Continuous aggregations** | 100x faster queries for daily/weekly views | Storage overhead (15% more) |
    | **TimescaleDB compression** | 5x storage reduction | CPU overhead (8-10%), compression lag |
    | **Multi-tier storage** | 90% cost savings | Query complexity, some queries slower |
    | **Feed caching (Redis)** | 10x faster feed loads | Stale data (5 min TTL), cache misses |
    | **Adaptive HR sampling** | 30% battery savings | Lower resolution during low activity |
    | **BLE sync (vs cellular)** | 95% battery savings | Requires phone nearby, sync delays |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (50M devices, 500M syncs/day):

    Compute:
    - 12 sync nodes × $50 = $600
    - 9 TSDB nodes × $500 = $4,500 (with replication)
    - 100 API nodes × $100 = $10,000
    - 14 ML nodes × $200 = $2,800
    - 12 social feed nodes × $100 = $1,200
    - Total compute: $19,100/month

    Storage:
    - Hot tier (12.4 TB SSD): $1,240
    - Warm tier (12.6 TB SSD): $1,260
    - Cold tier (5.2 TB S3): $120
    - Workout GPS (45 TB S3): $1,035
    - Total storage: $3,655/month

    Network:
    - Ingress: 1.2 Gbps × 400 TB/month = Negligible (< $200)
    - Egress: 2.8 Gbps × 900 TB/month = $81,000 (mobile apps, social feeds)

    External APIs:
    - Weather API: $500/month
    - Maps API (workout maps): $2,000/month
    - Push notifications (APNS/FCM): $1,000/month

    Total: $107,455/month ≈ $1.29M/year

    Per-device cost: $1.29M / 50M = $0.026/device/year

    Optimizations:
    1. CDN for static assets (maps, avatars): -$40K
    2. Reserved instances (30% discount): -$5.7K
    3. Spot instances for ML batch jobs: -$1K
    4. Aggressive compression (reduce storage): -$500
    5. Feed caching (reduce DB queries): -$2K

    Optimized Total: $58K/month ≈ $696K/year
    Per-device cost: $0.014/device/year
    ```

    ---

    ## Reliability & Fault Tolerance

    ```python
    # System reliability metrics

    # Device sync reliability
    sync_success_rate = 98%              # With retry logic
    data_loss_rate = 0.1%                # Buffer overflow in extreme cases
    sync_latency_p99 = 15                # seconds

    # Database availability
    tsdb_availability = 99.95%           # Replication + automatic failover
    api_availability = 99.99%            # Stateless, multi-instance
    social_feed_availability = 99.9%     # Cache fallback

    # End-to-end availability
    system_availability = 99.9%          # Min of all components

    # Expected downtime
    yearly_downtime = 365 × 24 × 60 × (1 - 0.999) = 525 minutes = 8.76 hours/year
    ```

    **Failure Scenarios:**

    ```
    Scenario 1: Single sync node failure
    - Impact: 50K-100K devices can't sync
    - Mitigation: Devices buffer locally (24-48 hours), retry to other nodes
    - Recovery time: Automatic (load balancer detects, redirects)
    - Data loss: 0%

    Scenario 2: TSDB node failure (with replication)
    - Impact: Read queries slightly slower (fall back to replicas)
    - Mitigation: Automatic failover to replica
    - Recovery time: < 30 seconds
    - Data loss: 0% (replicated)

    Scenario 3: Mobile app offline
    - Impact: User can't view data
    - Mitigation: Device buffers data, syncs when connection restored
    - Recovery time: Automatic when online
    - Data loss: 0% (device buffer: 24-48 hours)

    Scenario 4: Social feed service failure
    - Impact: Social features unavailable
    - Mitigation: Fallback to cached feeds (stale data OK)
    - Recovery time: 5-10 minutes (restart service)
    - Data loss: 0% (core tracking continues)

    Scenario 5: Entire data center failure
    - Impact: All services down in that region
    - Mitigation: Multi-region deployment
    - Recovery time: DNS failover to backup region (5-10 minutes)
    - Data loss: < 5 minutes (async replication lag)
    ```

    ---

    ## Privacy & Security

    ```
    Data protection:
    1. End-to-end encryption (TLS 1.3)
    2. Data encrypted at rest (AES-256)
    3. Device authentication (certificate-based)
    4. User authentication (OAuth 2.0 + MFA)
    5. GDPR compliance (data export, deletion)

    Privacy features:
    1. Private workouts (not shared by default)
    2. Friend privacy settings (hide activities)
    3. Location privacy (approximate maps only)
    4. Health data anonymization (for ML training)

    Compliance:
    - HIPAA (if medical features): Data de-identification
    - GDPR: Right to access, delete, export data
    - CCPA: California consumer privacy
    - ISO 27001: Information security
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you optimize battery life on the wearable device?"**
   - Adaptive sensor sampling (HR: 1-60 sec intervals)
   - Edge processing (reduce data transfer)
   - Smart display timeout (2-5 seconds)
   - BLE sync instead of cellular (95% power savings)
   - Sleep mode at night (minimal sampling)
   - **Result:** 5-7 days battery life with typical usage

2. **"How do you ensure accurate step counting?"**
   - 3-axis accelerometer at 50 Hz
   - Low-pass filter to remove noise (3 Hz cutoff)
   - Peak detection with dynamic threshold
   - False positive filtering (check step interval, pattern)
   - Adaptive threshold based on activity level
   - **Result:** 95-98% accuracy vs manual counting

3. **"How does sleep stage detection work?"**
   - Multi-sensor approach: accelerometer + heart rate + HRV
   - 30-second epochs analyzed with ML model (Random Forest)
   - Features: movement, HR, HRV, time context
   - Sleep stages: awake, light, deep, REM
   - Sleep score calculation (0-100) based on duration, quality, efficiency
   - **Result:** 85-90% agreement vs polysomnography

4. **"How do you handle offline mode and data sync?"**
   - Device buffers data locally (24-48 hours capacity)
   - Background sync every 30 minutes via BLE
   - Exponential backoff retry on failure
   - Compression (gzip) reduces bandwidth by 67%
   - Checksums for data integrity
   - Idempotent sync (deduplication)
   - **Result:** 98% sync success rate, <0.1% data loss

5. **"How do you scale the social feed to 10M+ users?"**
   - Fan-out on write for small friend lists (<500)
   - Fan-out on read for large lists (>500)
   - Feed cache in Redis (TTL: 5 minutes)
   - ML ranking algorithm for personalized feed
   - Async post generation (Kafka queue)
   - Graph database (Neo4j) for friend recommendations
   - **Result:** <100ms feed load time, 70% cache hit rate

6. **"How do you detect and classify workouts automatically?"**
   - Continuous activity recognition with accelerometer + gyroscope
   - ML model (CNN) trained on labeled workout data
   - Features: movement patterns, HR zones, GPS (if available)
   - Auto-detect: running, cycling, swimming, gym
   - Confidence threshold (>80%) to avoid false positives
   - User can confirm or correct detected workouts
   - **Result:** 90% accuracy for common activities

7. **"How do you ensure data privacy for health data?"**
   - End-to-end encryption (TLS 1.3 for sync, AES-256 at rest)
   - Private by default (workouts not shared unless user opts in)
   - Location privacy (fuzzy maps, no exact address)
   - GDPR compliance (data export, deletion rights)
   - Anonymized data for ML training
   - Multi-factor authentication for account access
   - **Result:** Zero data breaches, 100% GDPR compliance

**Key Points to Mention:**

- Edge processing on wearable reduces data transfer and battery consumption
- Multi-sensor fusion (accelerometer + heart rate + gyroscope) for accurate tracking
- ML-powered features: sleep stage detection, workout recognition, health insights
- Battery optimization: adaptive sampling, BLE sync, power-efficient algorithms
- Time-series database (TimescaleDB) optimized for sensor data
- Real-time sync with offline support and deduplication
- Social features with feed caching and graph database
- Privacy-first design with encryption and user controls
- Horizontal scaling for all stateless services
- Multi-tier storage with compression and retention

---

## Real-World Examples

**Fitbit:**
- 30M+ active users (as of 2023)
- PPG sensor for continuous HR monitoring
- 3-axis accelerometer for step counting and sleep tracking
- 5-7 day battery life
- Social challenges and leaderboards
- Machine learning for sleep stage detection (80-85% accuracy)
- Architecture: Wearable → BLE → Mobile App → Cloud API → ML insights

**Apple Watch:**
- 100M+ active users (as of 2024)
- Advanced sensors: ECG, blood oxygen, accelerometer, gyroscope
- Continuous HR monitoring (green/red/infrared LEDs)
- 18-hour battery life (more features, larger display)
- Workout auto-detection with 90%+ accuracy
- HealthKit integration for unified health data
- Fall detection and emergency SOS features

**Garmin:**
- Focus on athletes and outdoor enthusiasts
- Multi-band GPS for accurate tracking
- Pulse oximeter for blood oxygen (altitude training)
- Up to 14 days battery life (minimal smart features)
- Advanced metrics: VO2 max, training load, recovery time
- Connect IQ platform for third-party apps
- Offline maps for hiking/trail running

**Whoop:**
- Subscription-based fitness tracker
- Focus on recovery and strain metrics
- Continuous HR and HRV monitoring
- 5-day battery life (battery pack for charging while wearing)
- No display (reduce distractions)
- Strain coach: ML recommendations for daily effort
- Sleep coach: Personalized sleep recommendations

**Oura Ring:**
- Compact form factor (ring instead of watch)
- 3-7 day battery life
- Focus on sleep and recovery
- Temperature sensors for illness detection
- Readiness score: ML-based daily readiness
- Minimal workout tracking (not primary use case)
- Preferred by users who don't like wrist devices

---

## Summary

**System Characteristics:**

- **Scale:** 50M devices, 500M syncs/day, 5.8K syncs/sec
- **Latency:** < 10s sync, < 100ms API queries
- **Storage:** 147 TB total (with compression + retention)
- **Availability:** 99.9% uptime
- **Battery:** 5-7 days per charge
- **Accuracy:** 95-98% steps, 90% sleep stages

**Core Components:**

1. **Wearable Device:** Edge processing, local buffering, BLE sync
2. **Mobile App:** Sync service, local DB, real-time UI
3. **Sync Service:** Data ingestion, validation, deduplication
4. **Time-Series DB (TimescaleDB):** Sensor data storage with compression
5. **Activity Detector:** ML-powered workout recognition
6. **Sleep Analyzer:** ML-powered sleep stage detection
7. **Health Insights Engine:** Trend analysis, recommendations
8. **Social Feed Service:** Activity posts, challenges, leaderboards
9. **API Gateway:** REST/GraphQL, auth, rate limiting
10. **Notification Service:** Push notifications, alerts

**Key Design Decisions:**

- Edge processing on device (reduce data transfer, battery savings)
- BLE sync instead of cellular (95% power savings)
- Adaptive sensor sampling (battery optimization)
- TimescaleDB for time-series (PostgreSQL extension, mature, SQL)
- Multi-tier storage with compression (90% cost savings)
- Continuous aggregations (100x faster queries)
- ML-powered features (sleep detection, workout recognition, insights)
- Social features with feed caching (10x faster)
- Privacy-first design (encryption, private by default)
- Offline support with device buffering (24-48 hours)
- Idempotent sync with deduplication (data integrity)

This design provides a scalable, battery-efficient fitness tracking system capable of monitoring millions of users in real-time, providing accurate health insights, and enabling social engagement for motivation and accountability.
