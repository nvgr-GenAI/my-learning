# Design Location Tracking System

A real-time location tracking system that monitors and manages GPS coordinates from millions of mobile devices, enabling features like geofencing, location history, location sharing, and proximity detection.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐ Medium | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M active users, 10M location updates/sec, 500K concurrent geofences |
| **Key Challenges** | Real-time ingestion, geospatial indexing, battery optimization, privacy compliance |
| **Core Concepts** | Geohashing, geofencing algorithms, time-series optimization, location smoothing |
| **Companies** | Uber, Lyft, DoorDash, Find My (Apple), Google Maps, Life360, Tile |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Location Updates** | Receive real-time GPS coordinates from mobile devices | P0 (Must have) |
    | **Location History** | Store and query historical location trails | P0 (Must have) |
    | **Geofencing** | Trigger events when entering/exiting geographic zones | P0 (Must have) |
    | **Location Sharing** | Share live location with friends/family | P1 (Should have) |
    | **Proximity Detection** | Detect when users are near each other | P1 (Should have) |
    | **Location Query** | Get current location of user/device | P0 (Must have) |
    | **Trail Visualization** | Display movement path on map | P1 (Should have) |
    | **Privacy Controls** | Enable/disable tracking, set sharing permissions | P0 (Must have) |

    **Explicitly Out of Scope** (mention in interview):

    - Navigation and routing
    - Traffic prediction
    - Place search and discovery
    - Turn-by-turn directions
    - Indoor positioning

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Location tracking is critical for safety features |
    | **Latency (Updates)** | < 2s end-to-end | Real-time tracking requires fast processing |
    | **Latency (Queries)** | < 100ms | Fast location retrieval for responsive apps |
    | **Accuracy** | < 50m GPS accuracy | Sufficient for geofencing and location sharing |
    | **Scalability** | Handle 10M updates/sec | Support global user base |
    | **Privacy** | GDPR/CCPA compliant | Location is sensitive personal data |
    | **Battery Efficiency** | < 5% battery drain/hour | Mobile devices have limited battery |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total active users: 50M daily active users
    Tracking modes:
    - Active tracking (driving/moving): 10M users (20%)
    - Passive tracking (stationary): 40M users (80%)

    Location update frequency:
    - Active: every 4 seconds
    - Passive: every 30 seconds

    Updates per second:
    - Active: 10M users × 0.25 updates/sec = 2.5M updates/sec
    - Passive: 40M users × 0.033 updates/sec = 1.3M updates/sec
    - Total: 3.8M updates/sec average
    - Peak (3x): 11.4M updates/sec

    Location queries:
    - User checks location: 50M users × 10 checks/day = 500M queries/day
    - Average: 5,800 queries/sec
    - Peak: 17,400 queries/sec

    Geofence checks:
    - Active geofences: 500K concurrent
    - Per update: 3.8M updates/sec × 2 geofences avg = 7.6M checks/sec

    Location sharing:
    - Active shares: 5M concurrent
    - Update frequency: every 5 seconds
    - Bandwidth: 5M × 0.2 = 1M updates/sec
    ```

    ### Storage Estimates

    ```
    Location update data:
    - Per update: 60 bytes (user_id, lat, lng, timestamp, accuracy, speed, heading)
    - Daily: 3.8M updates/sec × 86,400 = 328B updates/day
    - Storage: 328B × 60 bytes = 19.7 TB/day
    - With compression (3x): 6.6 TB/day
    - 30-day retention: 198 TB

    Location history (aggregated):
    - Keep raw data for 7 days: 46 TB
    - Aggregate to 1-minute intervals: 0.5 TB
    - 1-year history: 6 TB

    Geofence definitions:
    - Per geofence: 500 bytes (id, user_id, center, radius, type)
    - Total geofences: 50M users × 5 geofences = 250M geofences
    - Storage: 250M × 500 bytes = 125 GB

    User metadata:
    - 50M users × 5 KB = 250 GB

    Total storage: 198 TB (raw locations) + 6 TB (history) + 0.375 TB (metadata) ≈ 204 TB
    ```

    ### Memory Estimates (Caching)

    ```
    Current location cache:
    - 50M users × 100 bytes = 5 GB

    Active geofences index:
    - 500K active × 1 KB = 500 MB

    Location sharing sessions:
    - 5M sessions × 500 bytes = 2.5 GB

    Geospatial index (Redis):
    - 50M locations in memory: 10 GB

    Total cache: 18 GB
    ```

    ---

    ## Key Assumptions

    1. Average user moves 20 km/day
    2. GPS accuracy: 10-50 meters (varies by environment)
    3. Mobile devices report location every 4-30 seconds
    4. 10% of users have location sharing enabled
    5. 1% of users have active geofences
    6. Battery consumption goal: < 5% per hour
    7. Location data retained for 30 days (raw), 1 year (aggregated)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Time-series optimization** - Location data is inherently time-ordered
    2. **Geospatial indexing** - Fast proximity and geofence queries
    3. **Write-heavy architecture** - 1000x more writes than reads
    4. **Client-side intelligence** - Battery optimization through adaptive updates
    5. **Privacy by design** - Encryption, access controls, data retention

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile Apps<br/>iOS/Android]
            Web[Web Dashboard]
        end

        subgraph "CDN & Load Balancing"
            CDN[CloudFront CDN]
            LB[Load Balancer<br/>ALB]
        end

        subgraph "API Gateway"
            API_GW[API Gateway<br/>Auth, Rate limiting]
            WS[WebSocket Servers<br/>Real-time updates]
        end

        subgraph "Core Services"
            Ingestion[Location Ingestion<br/>Service]
            Query[Location Query<br/>Service]
            Geofence[Geofence Service<br/>Event detection]
            Sharing[Location Sharing<br/>Service]
            History[Location History<br/>Service]
        end

        subgraph "Processing Layer"
            Stream[Stream Processing<br/>Apache Kafka]
            Processor[Location Processor<br/>Smoothing, validation]
        end

        subgraph "Geospatial"
            GeoIndex[Geospatial Index<br/>Redis + Geohash]
            Quadtree[QuadTree Index<br/>For dense areas]
        end

        subgraph "Storage"
            TimeSeries[(Time-Series DB<br/>InfluxDB/TimescaleDB<br/>Raw locations)]
            Cassandra[(Cassandra<br/>Historical data)]
            Postgres[(PostgreSQL<br/>User data, geofences)]
            S3[(S3<br/>Cold storage)]
        end

        subgraph "Cache Layer"
            Redis_Current[Redis<br/>Current locations]
            Redis_Geo[Redis<br/>Geospatial queries]
            Memcached[Memcached<br/>User sessions]
        end

        subgraph "Analytics & ML"
            Analytics[Analytics Service<br/>Movement patterns]
            ML[ML Service<br/>Battery optimization]
        end

        Mobile --> CDN
        Web --> CDN
        CDN --> LB
        LB --> API_GW
        Mobile -.->|WebSocket| WS

        API_GW --> Ingestion
        API_GW --> Query
        API_GW --> Geofence
        API_GW --> Sharing

        Ingestion --> Stream
        Stream --> Processor

        Processor --> TimeSeries
        Processor --> Redis_Current
        Processor --> GeoIndex
        Processor --> Geofence

        Query --> Redis_Current
        Query --> GeoIndex
        Query --> TimeSeries

        Geofence --> Redis_Geo
        Geofence --> Postgres
        Geofence --> Stream

        Sharing --> WS
        Sharing --> Redis_Current

        History --> Cassandra
        History --> S3

        Processor --> Analytics
        Analytics --> ML

        TimeSeries -.->|Archive| Cassandra
        Cassandra -.->|Archive| S3

        style LB fill:#e1f5ff
        style GeoIndex fill:#e8f5e9
        style Redis_Current fill:#fff4e1
        style Redis_Geo fill:#fff4e1
        style TimeSeries fill:#ffe1e1
        style Cassandra fill:#ffe1e1
        style Postgres fill:#ffe1e1
        style Stream fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **InfluxDB/TimescaleDB** | Optimized for time-series writes (3.8M/sec) | PostgreSQL (not optimized for time-series), MongoDB (poor compression) |
    | **Redis Geohash** | Sub-100ms geospatial queries | PostGIS (slower queries), Elasticsearch (higher overhead) |
    | **Apache Kafka** | Handle 10M+ events/sec with replay capability | RabbitMQ (lower throughput), direct writes (no buffering) |
    | **Cassandra** | Distributed storage for historical data (198 TB) | S3 (too slow for queries), PostgreSQL (can't handle scale) |
    | **WebSocket** | Real-time location sharing updates | HTTP polling (wasteful), Server-Sent Events (one-way only) |

    **Key Trade-off:** We chose **eventual consistency for location updates** but **strong consistency for geofence definitions**. Missing a location point is acceptable, but triggering geofence events incorrectly is not.

    ---

    ## API Design

    ### 1. Update Location

    **Request:**
    ```http
    POST /api/v1/locations
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "device_id": "device_abc123",
      "location": {
        "lat": 37.7749,
        "lng": -122.4194,
        "accuracy": 12.5,
        "altitude": 10.2,
        "speed": 5.5,
        "heading": 180
      },
      "timestamp": 1643712000000,
      "battery_level": 65,
      "motion_state": "moving"  // moving, stationary, driving
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "status": "accepted",
      "next_update_interval": 4,  // seconds
      "geofence_events": [
        {
          "geofence_id": "geo_xyz789",
          "event_type": "enter",
          "geofence_name": "Home"
        }
      ],
      "nearby_users": [
        {
          "user_id": "user_456",
          "distance_meters": 150,
          "shared_since": 1643711000000
        }
      ]
    }
    ```

    **Design Notes:**

    - 202 Accepted (async processing)
    - Server provides adaptive update interval based on motion state
    - Include geofence events in response to reduce client polling
    - Battery level helps server optimize update frequency

    ---

    ### 2. Get Current Location

    **Request:**
    ```http
    GET /api/v1/locations/user_123/current
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user_123",
      "location": {
        "lat": 37.7749,
        "lng": -122.4194,
        "accuracy": 15.0
      },
      "timestamp": 1643712000000,
      "last_seen": "2 minutes ago",
      "motion_state": "stationary"
    }
    ```

    ---

    ### 3. Create Geofence

    **Request:**
    ```http
    POST /api/v1/geofences
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "name": "Home",
      "center": {
        "lat": 37.7749,
        "lng": -122.4194
      },
      "radius_meters": 100,
      "trigger": {
        "on_enter": true,
        "on_exit": true,
        "on_dwell": {
          "enabled": true,
          "duration_seconds": 300
        }
      },
      "notification": {
        "enabled": true,
        "message": "Welcome home!"
      },
      "active_hours": {
        "start": "08:00",
        "end": "22:00",
        "timezone": "America/Los_Angeles"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "geofence_id": "geo_xyz789",
      "name": "Home",
      "status": "active",
      "created_at": 1643712000000
    }
    ```

    ---

    ### 4. Get Location History

    **Request:**
    ```http
    GET /api/v1/locations/user_123/history?start=1643700000000&end=1643712000000&interval=1m
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user_123",
      "start_time": 1643700000000,
      "end_time": 1643712000000,
      "interval": "1m",
      "points": [
        {
          "timestamp": 1643700000000,
          "lat": 37.7749,
          "lng": -122.4194,
          "accuracy": 20.0
        },
        {
          "timestamp": 1643700060000,
          "lat": 37.7750,
          "lng": -122.4195,
          "accuracy": 18.0
        }
        // ... more points
      ],
      "summary": {
        "total_points": 200,
        "distance_traveled_km": 5.2,
        "avg_speed_kmh": 45.0
      }
    }
    ```

    ---

    ### 5. Share Location

    **Request:**
    ```http
    POST /api/v1/sharing/sessions
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "recipients": ["user_456", "user_789"],
      "duration_minutes": 60,
      "precision": "approximate"  // exact, approximate, city-level
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "session_id": "share_abc123",
      "expires_at": 1643715600000,
      "websocket_url": "wss://ws.example.com/share/share_abc123"
    }
    ```

    ---

    ## Database Schema

    ### User Locations (TimescaleDB)

    ```sql
    -- Hypertable for time-series location data
    CREATE TABLE location_updates (
        user_id BIGINT NOT NULL,
        device_id VARCHAR(100),
        timestamp TIMESTAMPTZ NOT NULL,
        location GEOGRAPHY(POINT, 4326) NOT NULL,
        accuracy REAL,
        altitude REAL,
        speed REAL,
        heading SMALLINT,
        battery_level SMALLINT,
        motion_state VARCHAR(20),
        PRIMARY KEY (user_id, timestamp)
    );

    -- Convert to hypertable (TimescaleDB)
    SELECT create_hypertable('location_updates', 'timestamp');

    -- Automatically aggregate to 1-minute intervals
    CREATE MATERIALIZED VIEW location_updates_1min
    WITH (timescaledb.continuous) AS
    SELECT
        user_id,
        time_bucket('1 minute', timestamp) AS bucket,
        first(location, timestamp) as first_location,
        last(location, timestamp) as last_location,
        avg(accuracy) as avg_accuracy,
        count(*) as update_count
    FROM location_updates
    GROUP BY user_id, bucket;

    -- Retention policy: keep raw data for 7 days
    SELECT add_retention_policy('location_updates', INTERVAL '7 days');

    -- Compression policy
    SELECT add_compression_policy('location_updates', INTERVAL '1 day');
    ```

    ---

    ### Geofences (PostgreSQL)

    ```sql
    CREATE TABLE geofences (
        geofence_id UUID PRIMARY KEY,
        user_id BIGINT NOT NULL,
        name VARCHAR(200),
        center GEOGRAPHY(POINT, 4326) NOT NULL,
        radius_meters INT NOT NULL,
        trigger_on_enter BOOLEAN DEFAULT true,
        trigger_on_exit BOOLEAN DEFAULT true,
        trigger_on_dwell BOOLEAN DEFAULT false,
        dwell_duration_seconds INT,
        notification_enabled BOOLEAN DEFAULT true,
        notification_message TEXT,
        active_hours_start TIME,
        active_hours_end TIME,
        timezone VARCHAR(50),
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_geofences (user_id, is_active),
        INDEX idx_spatial (center) USING GIST  -- Spatial index
    );

    -- Geofence events log
    CREATE TABLE geofence_events (
        event_id BIGSERIAL PRIMARY KEY,
        geofence_id UUID NOT NULL,
        user_id BIGINT NOT NULL,
        event_type VARCHAR(20) NOT NULL,  -- enter, exit, dwell
        location GEOGRAPHY(POINT, 4326),
        timestamp TIMESTAMP NOT NULL,
        notification_sent BOOLEAN DEFAULT false,
        INDEX idx_user_events (user_id, timestamp),
        INDEX idx_geofence_events (geofence_id, timestamp)
    );
    ```

    ---

    ### Current Location Cache (Redis)

    ```
    # Current location (hash)
    HSET location:user:123 {
      "lat": 37.7749,
      "lng": -122.4194,
      "accuracy": 15.0,
      "timestamp": 1643712000000,
      "motion_state": "moving"
    }
    EXPIRE location:user:123 600  # 10 minute TTL

    # Geospatial index (for proximity queries)
    GEOADD locations:active -122.4194 37.7749 "user:123"

    # Location sharing session
    HSET sharing:session:abc123 {
      "owner_id": "user_123",
      "recipients": ["user_456", "user_789"],
      "expires_at": 1643715600000
    }
    EXPIRE sharing:session:abc123 3600
    ```

    ---

    ## Data Flow Diagrams

    ### Location Update Flow

    ```mermaid
    sequenceDiagram
        participant Mobile as Mobile App
        participant API as API Gateway
        participant Ingestion as Ingestion Service
        participant Kafka as Kafka Stream
        participant Processor as Location Processor
        participant Redis as Redis Cache
        participant TimeSeries as TimescaleDB
        participant Geofence as Geofence Service
        participant WS as WebSocket

        Mobile->>API: POST /locations (lat, lng, timestamp)
        API->>Ingestion: Forward location update

        Ingestion->>Ingestion: Validate coordinates
        Ingestion->>Kafka: Publish location event
        Ingestion-->>Mobile: 202 Accepted

        Kafka->>Processor: Consume location event

        par Parallel Processing
            Processor->>Processor: Smooth location (Kalman filter)
            Processor->>Redis: Update current location
            Processor->>Redis: GEOADD to spatial index
            Processor->>TimeSeries: Write time-series data
            Processor->>Geofence: Check geofence triggers
        end

        alt Geofence triggered
            Geofence->>Geofence: Detect enter/exit event
            Geofence->>Kafka: Publish geofence event
            Geofence->>Mobile: Push notification
        end

        alt Location sharing active
            Processor->>WS: Push to sharing subscribers
            WS->>Mobile: Real-time update
        end
    ```

    **Flow Explanation:**

    1. **Mobile sends location** - GPS coordinates, timestamp, metadata
    2. **Validation** - Check coordinate bounds, timestamp freshness
    3. **Async processing** - Publish to Kafka for reliability
    4. **Parallel writes** - Update cache, time-series DB, geospatial index
    5. **Geofence check** - Evaluate all active geofences for user
    6. **Event notification** - Trigger alerts, push notifications
    7. **Location sharing** - Push to active WebSocket connections

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical location tracking subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Geohashing & Indexing** | How to query nearby users in < 100ms? | Redis Geohash + QuadTree for dense areas |
    | **Geofencing Algorithm** | How to detect enter/exit events efficiently? | Ray casting + grid-based optimization |
    | **Location Smoothing** | How to handle GPS noise and jumps? | Kalman filter + outlier detection |
    | **Battery Optimization** | How to minimize battery drain? | Adaptive update intervals + motion detection |

    ---

    === "🗺️ Geospatial Indexing"

        ## The Challenge

        **Problem:** Find all users within 5km radius in < 100ms from 50M active users.

        **Naive approach:** Calculate distance to all 50M users. **Too slow:** O(N) = seconds.

        **Solution:** Geohash + Redis geospatial index

        ---

        ## Geohash Implementation

        **Concept:** Encode (lat, lng) as short string. Nearby locations share prefix.

        **Geohash precision:**

        | Length | Cell Size | Use Case |
        |--------|-----------|----------|
        | 3 | 156 km × 156 km | Country-level |
        | 4 | 39 km × 19 km | City-level |
        | 5 | 5 km × 5 km | Neighborhood |
        | 6 | 1.2 km × 0.6 km | **Location tracking** |
        | 7 | 153 m × 153 m | Street-level |
        | 8 | 38 m × 19 m | Building-level |

        **Implementation:**

        ```python
        import geohash2
        import redis
        from typing import List, Tuple
        import time

        class LocationIndex:
            """Geospatial indexing for location tracking"""

            def __init__(self, redis_client: redis.Redis):
                self.redis = redis_client
                self.precision = 6  # ~1km cells
                self.location_key = "locations:active"

            def update_location(
                self,
                user_id: str,
                lat: float,
                lng: float,
                metadata: dict = None
            ) -> None:
                """
                Update user location in geospatial index

                Args:
                    user_id: User identifier
                    lat: Latitude (-90 to 90)
                    lng: Longitude (-180 to 180)
                    metadata: Additional data (accuracy, speed, etc.)
                """
                # Validate coordinates
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    raise ValueError(f"Invalid coordinates: ({lat}, {lng})")

                # Redis GEOADD (native geospatial command)
                self.redis.geoadd(self.location_key, (lng, lat, f"user:{user_id}"))

                # Store metadata separately
                if metadata:
                    self.redis.hset(
                        f"location:user:{user_id}",
                        mapping={
                            "lat": str(lat),
                            "lng": str(lng),
                            "timestamp": str(time.time()),
                            **{k: str(v) for k, v in metadata.items()}
                        }
                    )
                    self.redis.expire(f"location:user:{user_id}", 600)  # 10 min TTL

                # Also maintain geohash index for grid queries
                ghash = geohash2.encode(lat, lng, precision=self.precision)
                self.redis.sadd(f"geohash:{ghash}", user_id)
                self.redis.expire(f"geohash:{ghash}", 3600)

            def find_nearby_users(
                self,
                lat: float,
                lng: float,
                radius_km: float = 5,
                limit: int = 100
            ) -> List[dict]:
                """
                Find users within radius using Redis GEORADIUS

                Args:
                    lat: Center latitude
                    lng: Center longitude
                    radius_km: Search radius in kilometers
                    limit: Maximum results

                Returns:
                    List of {user_id, distance_km, lat, lng}
                """
                # Use Redis GEORADIUS (O(N+log(M)) where N=results, M=total)
                results = self.redis.georadius(
                    self.location_key,
                    lng,
                    lat,
                    radius_km,
                    unit='km',
                    withdist=True,
                    withcoord=True,
                    count=limit,
                    sort='ASC'
                )

                nearby_users = []
                for member, distance, coords in results:
                    user_id = member.decode('utf-8').replace('user:', '')

                    # Get metadata
                    metadata = self.redis.hgetall(f"location:user:{user_id}")

                    nearby_users.append({
                        'user_id': user_id,
                        'distance_km': round(distance, 2),
                        'lat': coords[1],
                        'lng': coords[0],
                        'accuracy': float(metadata.get(b'accuracy', 0)),
                        'timestamp': int(float(metadata.get(b'timestamp', 0)))
                    })

                return nearby_users

            def find_in_bounding_box(
                self,
                min_lat: float,
                max_lat: float,
                min_lng: float,
                max_lng: float
            ) -> List[str]:
                """
                Find all users in bounding box (for geofencing)

                Args:
                    min_lat, max_lat, min_lng, max_lng: Box boundaries

                Returns:
                    List of user IDs
                """
                # Use Redis GEOSEARCH (Redis 6.2+)
                results = self.redis.geosearch(
                    self.location_key,
                    member=None,
                    longitude=(min_lng + max_lng) / 2,
                    latitude=(min_lat + max_lat) / 2,
                    radius=self._calculate_radius(min_lat, max_lat, min_lng, max_lng),
                    unit='km'
                )

                return [r.decode('utf-8').replace('user:', '') for r in results]

            def _calculate_radius(
                self,
                min_lat: float,
                max_lat: float,
                min_lng: float,
                max_lng: float
            ) -> float:
                """Calculate radius that encompasses bounding box"""
                from math import radians, cos, sin, asin, sqrt

                # Haversine distance for diagonal
                lat1, lng1 = radians(min_lat), radians(min_lng)
                lat2, lng2 = radians(max_lat), radians(max_lng)

                dlat = lat2 - lat1
                dlng = lng2 - lng1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
                c = 2 * asin(sqrt(a))
                return 6371 * c / 2  # Half diagonal
        ```

        **Performance:**

        - Redis GEORADIUS: O(N + log(M)) where N=results, M=total locations
        - Typical query: < 10ms for 50M locations
        - Memory: ~100 bytes per location = 5 GB for 50M users

        ---

        ## QuadTree for Dense Areas

        **Problem:** Dense urban areas (Manhattan) may have 100K users in single geohash cell.

        **Solution:** QuadTree for hierarchical spatial partitioning.

        ```python
        class QuadTree:
            """Hierarchical spatial index for dense areas"""

            def __init__(self, boundary, capacity=100):
                self.boundary = boundary  # (min_lat, max_lat, min_lng, max_lng)
                self.capacity = capacity
                self.points = []
                self.divided = False
                self.nw = self.ne = self.sw = self.se = None

            def insert(self, user_id: str, lat: float, lng: float) -> bool:
                """Insert user location into QuadTree"""
                if not self._in_boundary(lat, lng):
                    return False

                if len(self.points) < self.capacity:
                    self.points.append((user_id, lat, lng))
                    return True

                # Subdivide if at capacity
                if not self.divided:
                    self._subdivide()

                # Try to insert in subdivisions
                return (
                    self.nw.insert(user_id, lat, lng) or
                    self.ne.insert(user_id, lat, lng) or
                    self.sw.insert(user_id, lat, lng) or
                    self.se.insert(user_id, lat, lng)
                )

            def query_radius(self, lat: float, lng: float, radius_km: float) -> List[Tuple]:
                """Find all points within radius"""
                found = []

                # Check if circle intersects this quadrant
                if not self._intersects_circle(lat, lng, radius_km):
                    return found

                # Check points in this node
                for user_id, point_lat, point_lng in self.points:
                    if self._distance(lat, lng, point_lat, point_lng) <= radius_km:
                        found.append((user_id, point_lat, point_lng))

                # Recursively search children
                if self.divided:
                    found.extend(self.nw.query_radius(lat, lng, radius_km))
                    found.extend(self.ne.query_radius(lat, lng, radius_km))
                    found.extend(self.sw.query_radius(lat, lng, radius_km))
                    found.extend(self.se.query_radius(lat, lng, radius_km))

                return found

            def _subdivide(self):
                """Split into 4 quadrants"""
                min_lat, max_lat, min_lng, max_lng = self.boundary
                mid_lat = (min_lat + max_lat) / 2
                mid_lng = (min_lng + max_lng) / 2

                self.nw = QuadTree((mid_lat, max_lat, min_lng, mid_lng), self.capacity)
                self.ne = QuadTree((mid_lat, max_lat, mid_lng, max_lng), self.capacity)
                self.sw = QuadTree((min_lat, mid_lat, min_lng, mid_lng), self.capacity)
                self.se = QuadTree((min_lat, mid_lat, mid_lng, max_lng), self.capacity)

                self.divided = True

            def _in_boundary(self, lat: float, lng: float) -> bool:
                """Check if point is in this quadrant"""
                min_lat, max_lat, min_lng, max_lng = self.boundary
                return (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng)

            def _intersects_circle(self, lat: float, lng: float, radius_km: float) -> bool:
                """Check if circle intersects this quadrant boundary"""
                min_lat, max_lat, min_lng, max_lng = self.boundary

                # Find closest point on boundary to circle center
                closest_lat = max(min_lat, min(lat, max_lat))
                closest_lng = max(min_lng, min(lng, max_lng))

                # Check if closest point is within radius
                return self._distance(lat, lng, closest_lat, closest_lng) <= radius_km

            def _distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """Haversine distance in km"""
                from math import radians, cos, sin, asin, sqrt

                lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
                dlat = lat2 - lat1
                dlng = lng2 - lng1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
                c = 2 * asin(sqrt(a))
                return 6371 * c
        ```

    === "📍 Geofencing Algorithm"

        ## The Challenge

        **Problem:** Detect when user enters/exits geofence zone efficiently.

        **Requirements:**
        - Real-time detection (< 2 second latency)
        - Support 500K concurrent geofences
        - Handle 3.8M location updates/sec
        - Minimize false positives/negatives

        ---

        ## Geofence Detection Algorithm

        **Approach:** Combine spatial indexing + state machine

        ```python
        from enum import Enum
        from typing import List, Optional
        import time

        class GeofenceEventType(Enum):
            ENTER = "enter"
            EXIT = "exit"
            DWELL = "dwell"

        class GeofenceState(Enum):
            OUTSIDE = "outside"
            INSIDE = "inside"
            DWELLING = "dwelling"

        class GeofenceEngine:
            """Detect geofence enter/exit/dwell events"""

            def __init__(self, redis_client, db_connection):
                self.redis = redis_client
                self.db = db_connection
                self.dwell_threshold = 300  # 5 minutes

            def check_geofences(
                self,
                user_id: str,
                lat: float,
                lng: float,
                timestamp: int
            ) -> List[dict]:
                """
                Check all geofences for user and detect events

                Args:
                    user_id: User identifier
                    lat: Current latitude
                    lng: Current longitude
                    timestamp: Unix timestamp

                Returns:
                    List of geofence events
                """
                events = []

                # Get all active geofences for user
                geofences = self._get_user_geofences(user_id)

                for geofence in geofences:
                    # Check if location is inside geofence
                    is_inside = self._point_in_circle(
                        lat,
                        lng,
                        geofence['center_lat'],
                        geofence['center_lng'],
                        geofence['radius_meters']
                    )

                    # Get previous state
                    prev_state = self._get_geofence_state(user_id, geofence['geofence_id'])

                    # Detect state changes
                    if is_inside and prev_state == GeofenceState.OUTSIDE:
                        # ENTER event
                        if geofence['trigger_on_enter']:
                            events.append({
                                'geofence_id': geofence['geofence_id'],
                                'event_type': GeofenceEventType.ENTER,
                                'geofence_name': geofence['name'],
                                'timestamp': timestamp
                            })

                        # Update state
                        self._set_geofence_state(
                            user_id,
                            geofence['geofence_id'],
                            GeofenceState.INSIDE,
                            timestamp
                        )

                    elif not is_inside and prev_state in [GeofenceState.INSIDE, GeofenceState.DWELLING]:
                        # EXIT event
                        if geofence['trigger_on_exit']:
                            events.append({
                                'geofence_id': geofence['geofence_id'],
                                'event_type': GeofenceEventType.EXIT,
                                'geofence_name': geofence['name'],
                                'timestamp': timestamp
                            })

                        # Update state
                        self._set_geofence_state(
                            user_id,
                            geofence['geofence_id'],
                            GeofenceState.OUTSIDE,
                            timestamp
                        )

                    elif is_inside and prev_state == GeofenceState.INSIDE:
                        # Check for DWELL
                        if geofence['trigger_on_dwell']:
                            entry_time = self._get_entry_timestamp(
                                user_id,
                                geofence['geofence_id']
                            )

                            if entry_time and (timestamp - entry_time) >= geofence['dwell_duration_seconds']:
                                events.append({
                                    'geofence_id': geofence['geofence_id'],
                                    'event_type': GeofenceEventType.DWELL,
                                    'geofence_name': geofence['name'],
                                    'timestamp': timestamp,
                                    'duration_seconds': timestamp - entry_time
                                })

                                # Update to dwelling state
                                self._set_geofence_state(
                                    user_id,
                                    geofence['geofence_id'],
                                    GeofenceState.DWELLING,
                                    timestamp
                                )

                # Log events
                for event in events:
                    self._log_geofence_event(user_id, event)

                return events

            def _get_user_geofences(self, user_id: str) -> List[dict]:
                """Get all active geofences for user"""
                # Check cache first
                cache_key = f"geofences:user:{user_id}"
                cached = self.redis.get(cache_key)

                if cached:
                    import json
                    return json.loads(cached)

                # Query database
                query = """
                    SELECT geofence_id, name,
                           ST_Y(center::geometry) as center_lat,
                           ST_X(center::geometry) as center_lng,
                           radius_meters,
                           trigger_on_enter, trigger_on_exit, trigger_on_dwell,
                           dwell_duration_seconds
                    FROM geofences
                    WHERE user_id = %s AND is_active = true
                """

                geofences = self.db.execute(query, [user_id])

                # Cache for 5 minutes
                self.redis.setex(cache_key, 300, json.dumps(geofences))

                return geofences

            def _point_in_circle(
                self,
                point_lat: float,
                point_lng: float,
                center_lat: float,
                center_lng: float,
                radius_meters: float
            ) -> bool:
                """Check if point is inside circular geofence"""
                from math import radians, cos, sin, asin, sqrt

                # Haversine distance
                lat1, lng1, lat2, lng2 = map(
                    radians,
                    [point_lat, point_lng, center_lat, center_lng]
                )

                dlat = lat2 - lat1
                dlng = lng2 - lng1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
                c = 2 * asin(sqrt(a))
                distance_meters = 6371000 * c  # Earth radius in meters

                return distance_meters <= radius_meters

            def _get_geofence_state(self, user_id: str, geofence_id: str) -> GeofenceState:
                """Get current geofence state for user"""
                state_key = f"geofence:state:{user_id}:{geofence_id}"
                state = self.redis.hget(state_key, "state")

                if not state:
                    return GeofenceState.OUTSIDE

                return GeofenceState(state.decode('utf-8'))

            def _set_geofence_state(
                self,
                user_id: str,
                geofence_id: str,
                state: GeofenceState,
                timestamp: int
            ) -> None:
                """Update geofence state"""
                state_key = f"geofence:state:{user_id}:{geofence_id}"

                self.redis.hset(state_key, mapping={
                    "state": state.value,
                    "timestamp": str(timestamp)
                })

                self.redis.expire(state_key, 86400)  # 24 hour TTL

            def _get_entry_timestamp(self, user_id: str, geofence_id: str) -> Optional[int]:
                """Get timestamp when user entered geofence"""
                state_key = f"geofence:state:{user_id}:{geofence_id}"
                timestamp = self.redis.hget(state_key, "timestamp")

                if timestamp:
                    return int(timestamp.decode('utf-8'))
                return None

            def _log_geofence_event(self, user_id: str, event: dict) -> None:
                """Log geofence event to database"""
                query = """
                    INSERT INTO geofence_events
                    (geofence_id, user_id, event_type, timestamp, notification_sent)
                    VALUES (%s, %s, %s, to_timestamp(%s), false)
                """

                self.db.execute(query, [
                    event['geofence_id'],
                    user_id,
                    event['event_type'].value,
                    event['timestamp']
                ])
        ```

        ---

        ## Optimization: Grid-Based Indexing

        **Problem:** Checking all geofences for every location update is expensive.

        **Solution:** Spatial grid index to quickly find relevant geofences.

        ```python
        class GeofenceGrid:
            """Grid-based spatial index for geofences"""

            def __init__(self, cell_size_km=1.0):
                self.cell_size_km = cell_size_km
                self.grid = {}  # {(cell_x, cell_y): [geofence_ids]}

            def index_geofence(self, geofence: dict) -> None:
                """Add geofence to spatial grid"""
                center_lat = geofence['center_lat']
                center_lng = geofence['center_lng']
                radius_km = geofence['radius_meters'] / 1000.0

                # Find all grid cells that geofence intersects
                cells = self._get_intersecting_cells(
                    center_lat,
                    center_lng,
                    radius_km
                )

                for cell in cells:
                    if cell not in self.grid:
                        self.grid[cell] = []
                    self.grid[cell].append(geofence['geofence_id'])

            def get_nearby_geofences(self, lat: float, lng: float) -> List[str]:
                """Get geofences that might contain this location"""
                cell = self._get_cell(lat, lng)

                # Check this cell and 8 neighbors
                nearby_cells = [
                    cell,
                    (cell[0]-1, cell[1]-1), (cell[0], cell[1]-1), (cell[0]+1, cell[1]-1),
                    (cell[0]-1, cell[1]),                          (cell[0]+1, cell[1]),
                    (cell[0]-1, cell[1]+1), (cell[0], cell[1]+1), (cell[0]+1, cell[1]+1)
                ]

                geofence_ids = set()
                for c in nearby_cells:
                    if c in self.grid:
                        geofence_ids.update(self.grid[c])

                return list(geofence_ids)

            def _get_cell(self, lat: float, lng: float) -> tuple:
                """Convert lat/lng to grid cell coordinates"""
                cell_x = int(lng / (self.cell_size_km / 111.0))  # ~111 km per degree
                cell_y = int(lat / (self.cell_size_km / 111.0))
                return (cell_x, cell_y)

            def _get_intersecting_cells(
                self,
                center_lat: float,
                center_lng: float,
                radius_km: float
            ) -> List[tuple]:
                """Get all grid cells that geofence circle intersects"""
                # Calculate bounding box
                lat_delta = radius_km / 111.0
                lng_delta = radius_km / (111.0 * abs(math.cos(math.radians(center_lat))))

                min_lat = center_lat - lat_delta
                max_lat = center_lat + lat_delta
                min_lng = center_lng - lng_delta
                max_lng = center_lng + lng_delta

                # Get all cells in bounding box
                cells = []
                min_cell = self._get_cell(min_lat, min_lng)
                max_cell = self._get_cell(max_lat, max_lng)

                for x in range(min_cell[0], max_cell[0] + 1):
                    for y in range(min_cell[1], max_cell[1] + 1):
                        cells.append((x, y))

                return cells
        ```

        **Performance Improvement:**
        - Without grid: Check all 500K geofences = O(N)
        - With grid: Check only ~100 geofences per cell = O(1)
        - 5000x speedup

    === "📈 Location Smoothing"

        ## The Challenge

        **Problem:** GPS coordinates have noise and occasional large errors (jumps).

        **Examples:**
        - Random jitter: ±10-50 meters
        - GPS jumps: Sudden 200m+ errors
        - Signal loss: Missing data points
        - Multipath errors: Reflections in urban canyons

        **Solution:** Kalman filter + outlier detection

        ---

        ## Kalman Filter Implementation

        ```python
        import numpy as np
        from typing import Tuple, Optional

        class LocationSmoother:
            """Smooth GPS coordinates using Kalman filter"""

            def __init__(self):
                # Kalman filter state
                self.state = None  # [lat, lng, velocity_lat, velocity_lng]
                self.covariance = None

                # Process noise (how much we expect location to change)
                self.process_noise = np.array([
                    [0.001, 0, 0, 0],
                    [0, 0.001, 0, 0],
                    [0, 0, 0.01, 0],
                    [0, 0, 0, 0.01]
                ])

                # Measurement noise (GPS accuracy)
                self.measurement_noise = np.array([
                    [0.0001, 0],
                    [0, 0.0001]
                ])

            def smooth_location(
                self,
                lat: float,
                lng: float,
                accuracy: float,
                timestamp: int,
                speed: Optional[float] = None
            ) -> Tuple[float, float]:
                """
                Apply Kalman filter to smooth GPS coordinates

                Args:
                    lat: Raw latitude
                    lng: Raw longitude
                    accuracy: GPS accuracy in meters
                    timestamp: Unix timestamp
                    speed: Speed in m/s (if available)

                Returns:
                    (smoothed_lat, smoothed_lng)
                """
                # Initialize state if first measurement
                if self.state is None:
                    self.state = np.array([lat, lng, 0, 0])
                    self.covariance = np.eye(4) * 0.1
                    self.last_timestamp = timestamp
                    return lat, lng

                # Calculate time delta
                dt = timestamp - self.last_timestamp
                if dt <= 0:
                    return self.state[0], self.state[1]

                # Outlier detection
                predicted_lat, predicted_lng = self._predict(dt)
                distance = self._haversine_distance(
                    lat, lng,
                    predicted_lat, predicted_lng
                )

                # Reject obvious outliers (> 1km jump in 4 seconds = 250 m/s = 900 km/h)
                max_speed_mps = 100  # 360 km/h (driving max)
                max_distance = max_speed_mps * dt

                if distance > max_distance:
                    # Likely GPS error - use prediction instead
                    print(f"Outlier detected: {distance}m jump in {dt}s")
                    return predicted_lat, predicted_lng

                # Prediction step
                self._predict_step(dt)

                # Update step (incorporate measurement)
                measurement = np.array([lat, lng])

                # Adjust measurement noise based on GPS accuracy
                measurement_noise = np.eye(2) * (accuracy / 100000.0) ** 2

                # Kalman gain
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
                S = H @ self.covariance @ H.T + measurement_noise
                K = self.covariance @ H.T @ np.linalg.inv(S)

                # Update state
                innovation = measurement - (H @ self.state)
                self.state = self.state + K @ innovation
                self.covariance = (np.eye(4) - K @ H) @ self.covariance

                self.last_timestamp = timestamp

                return self.state[0], self.state[1]

            def _predict_step(self, dt: float) -> None:
                """Predict next state based on velocity"""
                # State transition matrix (constant velocity model)
                F = np.array([
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])

                # Predict state
                self.state = F @ self.state

                # Predict covariance
                self.covariance = F @ self.covariance @ F.T + self.process_noise

            def _predict(self, dt: float) -> Tuple[float, float]:
                """Predict location without updating state"""
                predicted_lat = self.state[0] + self.state[2] * dt
                predicted_lng = self.state[1] + self.state[3] * dt
                return predicted_lat, predicted_lng

            def _haversine_distance(
                self,
                lat1: float,
                lng1: float,
                lat2: float,
                lng2: float
            ) -> float:
                """Calculate distance in meters"""
                from math import radians, cos, sin, asin, sqrt

                lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
                dlat = lat2 - lat1
                dlng = lng2 - lng1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
                c = 2 * asin(sqrt(a))
                return 6371000 * c  # meters
        ```

        ---

        ## Snap to Road

        **Enhancement:** Snap GPS points to road network for driving scenarios.

        ```python
        class RoadSnapping:
            """Snap GPS coordinates to road network"""

            def __init__(self, road_network):
                self.road_network = road_network  # Graph of roads

            def snap_to_road(
                self,
                lat: float,
                lng: float,
                motion_state: str
            ) -> Tuple[float, float]:
                """
                Snap location to nearest road if driving

                Args:
                    lat: GPS latitude
                    lng: GPS longitude
                    motion_state: "driving", "walking", "stationary"

                Returns:
                    (snapped_lat, snapped_lng)
                """
                if motion_state != "driving":
                    return lat, lng

                # Find nearest road segment
                nearest_road = self._find_nearest_road(lat, lng, max_distance=50)

                if nearest_road:
                    # Project point onto road segment
                    snapped_lat, snapped_lng = self._project_to_segment(
                        lat, lng,
                        nearest_road['start_lat'], nearest_road['start_lng'],
                        nearest_road['end_lat'], nearest_road['end_lng']
                    )
                    return snapped_lat, snapped_lng

                return lat, lng

            def _find_nearest_road(
                self,
                lat: float,
                lng: float,
                max_distance: float
            ) -> Optional[dict]:
                """Find nearest road segment within max_distance meters"""
                # Query road network (using spatial index)
                nearby_roads = self.road_network.query_radius(lat, lng, max_distance)

                if not nearby_roads:
                    return None

                # Find closest road
                min_distance = float('inf')
                nearest_road = None

                for road in nearby_roads:
                    distance = self._distance_to_segment(
                        lat, lng,
                        road['start_lat'], road['start_lng'],
                        road['end_lat'], road['end_lng']
                    )

                    if distance < min_distance:
                        min_distance = distance
                        nearest_road = road

                return nearest_road

            def _project_to_segment(
                self,
                point_lat: float,
                point_lng: float,
                seg_start_lat: float,
                seg_start_lng: float,
                seg_end_lat: float,
                seg_end_lng: float
            ) -> Tuple[float, float]:
                """Project point onto line segment"""
                # Vector from start to end
                dx = seg_end_lat - seg_start_lat
                dy = seg_end_lng - seg_start_lng

                # Vector from start to point
                px = point_lat - seg_start_lat
                py = point_lng - seg_start_lng

                # Project onto segment
                t = (px * dx + py * dy) / (dx * dx + dy * dy)
                t = max(0, min(1, t))  # Clamp to [0, 1]

                projected_lat = seg_start_lat + t * dx
                projected_lng = seg_start_lng + t * dy

                return projected_lat, projected_lng

            def _distance_to_segment(
                self,
                point_lat: float,
                point_lng: float,
                seg_start_lat: float,
                seg_start_lng: float,
                seg_end_lat: float,
                seg_end_lng: float
            ) -> float:
                """Calculate distance from point to line segment"""
                projected_lat, projected_lng = self._project_to_segment(
                    point_lat, point_lng,
                    seg_start_lat, seg_start_lng,
                    seg_end_lat, seg_end_lng
                )

                return self._haversine_distance(
                    point_lat, point_lng,
                    projected_lat, projected_lng
                )
        ```

    === "🔋 Battery Optimization"

        ## The Challenge

        **Problem:** Continuous GPS tracking drains battery quickly (20% per hour).

        **Goal:** Reduce to < 5% per hour while maintaining tracking quality.

        **Solutions:**
        1. Adaptive update intervals
        2. Motion detection (accelerometer)
        3. Geofence-based tracking
        4. Network-based location (WiFi, cell towers)

        ---

        ## Adaptive Update Intervals

        ```python
        from enum import Enum
        from typing import Optional

        class MotionState(Enum):
            STATIONARY = "stationary"
            WALKING = "walking"
            DRIVING = "driving"
            HIGH_SPEED = "high_speed"  # Train, airplane

        class AdaptiveLocationManager:
            """Optimize location update frequency based on motion state"""

            # Update intervals (seconds)
            INTERVALS = {
                MotionState.STATIONARY: 60,      # 1 minute
                MotionState.WALKING: 15,         # 15 seconds
                MotionState.DRIVING: 4,          # 4 seconds
                MotionState.HIGH_SPEED: 10       # 10 seconds
            }

            # Battery thresholds
            LOW_BATTERY_THRESHOLD = 20  # %
            CRITICAL_BATTERY_THRESHOLD = 10  # %

            def __init__(self):
                self.current_state = MotionState.STATIONARY
                self.last_location = None
                self.last_update_time = 0
                self.distance_threshold = 50  # meters

            def get_next_update_interval(
                self,
                motion_state: MotionState,
                battery_level: int,
                distance_from_last: float,
                has_active_geofences: bool
            ) -> int:
                """
                Calculate optimal update interval

                Args:
                    motion_state: Current motion state
                    battery_level: Battery percentage (0-100)
                    distance_from_last: Distance from last update (meters)
                    has_active_geofences: Whether user has active geofences

                Returns:
                    Next update interval in seconds
                """
                # Base interval from motion state
                interval = self.INTERVALS[motion_state]

                # Battery conservation mode
                if battery_level <= self.CRITICAL_BATTERY_THRESHOLD:
                    # Critical battery - reduce frequency dramatically
                    interval *= 4
                elif battery_level <= self.LOW_BATTERY_THRESHOLD:
                    # Low battery - reduce frequency moderately
                    interval *= 2

                # If stationary, extend interval further
                if motion_state == MotionState.STATIONARY:
                    if distance_from_last < 10:  # Hasn't moved much
                        interval = min(interval * 2, 300)  # Max 5 minutes

                # If has active geofences, maintain higher frequency
                if has_active_geofences and motion_state in [MotionState.WALKING, MotionState.DRIVING]:
                    interval = max(interval // 2, 4)  # Min 4 seconds

                return interval

            def detect_motion_state(
                self,
                accelerometer_data: dict,
                gps_speed: Optional[float] = None,
                distance_from_last: float = 0
            ) -> MotionState:
                """
                Detect motion state using sensors

                Args:
                    accelerometer_data: {x, y, z} acceleration
                    gps_speed: Speed from GPS (m/s)
                    distance_from_last: Distance since last update (meters)

                Returns:
                    Detected motion state
                """
                # Calculate acceleration magnitude
                ax = accelerometer_data.get('x', 0)
                ay = accelerometer_data.get('y', 0)
                az = accelerometer_data.get('z', 0)

                acceleration_magnitude = (ax**2 + ay**2 + az**2) ** 0.5

                # Stationary threshold (gravity ~9.8 m/s^2 ± 0.5)
                if 9.3 < acceleration_magnitude < 10.3 and distance_from_last < 5:
                    return MotionState.STATIONARY

                # Use GPS speed if available
                if gps_speed is not None:
                    if gps_speed < 0.5:  # < 1.8 km/h
                        return MotionState.STATIONARY
                    elif gps_speed < 5:  # < 18 km/h
                        return MotionState.WALKING
                    elif gps_speed < 40:  # < 144 km/h
                        return MotionState.DRIVING
                    else:
                        return MotionState.HIGH_SPEED

                # Fallback to distance-based detection
                if distance_from_last < 5:
                    return MotionState.STATIONARY
                elif distance_from_last < 50:
                    return MotionState.WALKING
                else:
                    return MotionState.DRIVING
        ```

        ---

        ## Network-Based Location (WiFi/Cell)

        ```python
        class HybridLocationProvider:
            """Use multiple location sources to minimize battery drain"""

            def __init__(self):
                self.gps_enabled = True
                self.wifi_enabled = True
                self.cell_enabled = True

            def get_location(
                self,
                battery_level: int,
                accuracy_required: float,
                motion_state: MotionState
            ) -> dict:
                """
                Get location using most efficient method

                Battery consumption:
                - GPS: 40-50 mW
                - WiFi: 10-15 mW
                - Cell tower: 5-10 mW

                Accuracy:
                - GPS: 5-50m
                - WiFi: 20-100m
                - Cell tower: 100-2000m

                Args:
                    battery_level: Current battery percentage
                    accuracy_required: Required accuracy in meters
                    motion_state: Current motion state

                Returns:
                    {lat, lng, accuracy, source}
                """
                # Low battery mode - prefer network location
                if battery_level < self.CRITICAL_BATTERY_THRESHOLD:
                    if self.cell_enabled:
                        return self._get_cell_location()
                    elif self.wifi_enabled:
                        return self._get_wifi_location()

                # Stationary - use WiFi (more accurate than cell, cheaper than GPS)
                if motion_state == MotionState.STATIONARY:
                    if self.wifi_enabled and accuracy_required <= 100:
                        return self._get_wifi_location()

                # High accuracy needed (geofencing) - use GPS
                if accuracy_required < 50:
                    if self.gps_enabled:
                        return self._get_gps_location()

                # Default to WiFi if available
                if self.wifi_enabled:
                    return self._get_wifi_location()

                # Fallback to GPS
                return self._get_gps_location()

            def _get_gps_location(self) -> dict:
                """Get GPS location (high accuracy, high battery)"""
                # Request location from GPS hardware
                location = self._request_gps()

                return {
                    'lat': location['latitude'],
                    'lng': location['longitude'],
                    'accuracy': location['accuracy'],
                    'source': 'gps',
                    'battery_cost': 'high'
                }

            def _get_wifi_location(self) -> dict:
                """Get WiFi-based location (medium accuracy, low battery)"""
                # Scan nearby WiFi access points
                access_points = self._scan_wifi()

                # Send to location service for triangulation
                location = self._triangulate_wifi(access_points)

                return {
                    'lat': location['latitude'],
                    'lng': location['longitude'],
                    'accuracy': 50.0,  # ~50m accuracy
                    'source': 'wifi',
                    'battery_cost': 'low'
                }

            def _get_cell_location(self) -> dict:
                """Get cell tower location (low accuracy, very low battery)"""
                # Get connected cell tower
                cell_tower = self._get_cell_tower_info()

                # Lookup tower location
                location = self._lookup_cell_tower(cell_tower)

                return {
                    'lat': location['latitude'],
                    'lng': location['longitude'],
                    'accuracy': 500.0,  # ~500m accuracy
                    'source': 'cell',
                    'battery_cost': 'very_low'
                }
        ```

        ---

        ## Battery Optimization Summary

        **Strategies:**

        | Scenario | Update Interval | Location Source | Battery Impact |
        |----------|----------------|----------------|----------------|
        | Stationary, normal battery | 60s | WiFi | 1-2% per hour |
        | Walking, normal battery | 15s | GPS | 3-5% per hour |
        | Driving, normal battery | 4s | GPS | 8-10% per hour |
        | Stationary, low battery | 120s | Cell tower | < 1% per hour |
        | Walking, low battery | 30s | WiFi | 1-2% per hour |
        | Driving, low battery | 8s | GPS | 5-6% per hour |

        **Power Consumption Reduction:**
        - Without optimization: 20% per hour (continuous GPS)
        - With optimization: 3-5% per hour (adaptive)
        - Improvement: 75% reduction

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling location tracking from 1M to 50M users.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Location ingestion** | ✅ Yes | Kafka partitioning (1000 partitions), multiple consumers |
    | **Time-series writes** | ✅ Yes | InfluxDB cluster (50 nodes), batch writes (100x reduction) |
    | **Geospatial queries** | ✅ Yes | Redis cluster (20 nodes), read replicas |
    | **Geofence checks** | ✅ Yes | Grid-based indexing, parallel processing |
    | **Storage** | ✅ Yes | Data compression (3x), tiered storage (hot/warm/cold) |

    ---

    ## Scalability Solutions

    ### Write Path Optimization

    ```python
    class LocationBatcher:
        """Batch location updates for efficient writes"""

        def __init__(self, influxdb_client, batch_size=1000, flush_interval=1.0):
            self.influxdb = influxdb_client
            self.batch_size = batch_size
            self.flush_interval = flush_interval
            self.buffer = []
            self.last_flush = time.time()

        async def add_location(self, user_id: str, location: dict):
            """Add location to batch"""
            point = {
                "measurement": "location_updates",
                "tags": {
                    "user_id": user_id,
                    "motion_state": location.get('motion_state', 'unknown')
                },
                "fields": {
                    "lat": location['lat'],
                    "lng": location['lng'],
                    "accuracy": location.get('accuracy', 0),
                    "speed": location.get('speed', 0),
                    "heading": location.get('heading', 0)
                },
                "time": location['timestamp']
            }

            self.buffer.append(point)

            # Flush if batch full or interval elapsed
            if len(self.buffer) >= self.batch_size or \
               time.time() - self.last_flush >= self.flush_interval:
                await self.flush()

        async def flush(self):
            """Write buffered locations to InfluxDB"""
            if not self.buffer:
                return

            try:
                # Batch write
                self.influxdb.write_points(
                    self.buffer,
                    batch_size=self.batch_size,
                    protocol='line'
                )

                self.buffer = []
                self.last_flush = time.time()

            except Exception as e:
                logger.error(f"Failed to write batch: {e}")
                # Retry or dead-letter queue
    ```

    **Performance:**
    - Without batching: 3.8M writes/sec × 10ms = 38,000 seconds
    - With batching (1000x): 3,800 writes/sec × 10ms = 38 seconds
    - 1000x improvement

    ---

    ## Data Retention & Archival

    ```python
    class LocationArchiver:
        """Tiered storage strategy for location data"""

        def __init__(self):
            self.hot_retention_days = 7      # InfluxDB
            self.warm_retention_days = 30    # Cassandra
            self.cold_retention_years = 1    # S3

        def archive_old_data(self):
            """Move data through storage tiers"""

            # Hot -> Warm (7 days)
            # Aggregate to 1-minute intervals, move to Cassandra
            seven_days_ago = time.time() - (7 * 86400)

            query = f"""
                SELECT
                    user_id,
                    time_bucket('1 minute', timestamp) AS bucket,
                    first(location) AS start_location,
                    last(location) AS end_location,
                    avg(accuracy) AS avg_accuracy,
                    count(*) AS point_count
                FROM location_updates
                WHERE timestamp < {seven_days_ago}
                GROUP BY user_id, bucket
            """

            aggregated_data = self.influxdb.query(query)
            self.cassandra.insert_batch(aggregated_data)

            # Delete from InfluxDB
            self.influxdb.delete(f"timestamp < {seven_days_ago}")

            # Warm -> Cold (30 days)
            # Further aggregate to 1-hour intervals, move to S3
            thirty_days_ago = time.time() - (30 * 86400)

            hourly_data = self.cassandra.aggregate_to_hourly(thirty_days_ago)
            self.s3.upload_parquet(hourly_data, f"locations/{thirty_days_ago}.parquet")

            # Delete from Cassandra
            self.cassandra.delete(f"timestamp < {thirty_days_ago}")
    ```

    **Storage Optimization:**

    | Tier | Retention | Granularity | Storage | Query Speed |
    |------|-----------|-------------|---------|-------------|
    | Hot (InfluxDB) | 7 days | Raw (4s) | 46 TB | < 100ms |
    | Warm (Cassandra) | 30 days | 1-minute | 6 TB | < 1s |
    | Cold (S3) | 1 year | 1-hour | 0.5 TB | 5-10s |
    | **Total** | | | **52.5 TB** | |

    Without tiering: 204 TB (4x more expensive)

    ---

    ## Privacy & Security

    ### Data Encryption

    ```python
    from cryptography.fernet import Fernet
    import hashlib

    class LocationPrivacy:
        """Privacy-preserving location storage"""

        def __init__(self, master_key: bytes):
            self.master_key = master_key

        def encrypt_location(
            self,
            user_id: str,
            lat: float,
            lng: float
        ) -> dict:
            """Encrypt precise coordinates"""
            # Generate user-specific key
            user_key = self._derive_user_key(user_id)
            cipher = Fernet(user_key)

            # Encrypt coordinates
            location_data = f"{lat},{lng}".encode()
            encrypted = cipher.encrypt(location_data)

            # Store encrypted location with geohash for queries
            geohash = geohash2.encode(lat, lng, precision=4)  # City-level

            return {
                'user_id': user_id,
                'encrypted_location': encrypted.decode(),
                'geohash_prefix': geohash,  # For proximity queries
                'timestamp': time.time()
            }

        def decrypt_location(
            self,
            user_id: str,
            encrypted_location: str
        ) -> Tuple[float, float]:
            """Decrypt coordinates"""
            user_key = self._derive_user_key(user_id)
            cipher = Fernet(user_key)

            decrypted = cipher.decrypt(encrypted_location.encode())
            lat_str, lng_str = decrypted.decode().split(',')

            return float(lat_str), float(lng_str)

        def _derive_user_key(self, user_id: str) -> bytes:
            """Derive encryption key for user"""
            key_material = hashlib.pbkdf2_hmac(
                'sha256',
                user_id.encode(),
                self.master_key,
                100000
            )
            return base64.urlsafe_b64encode(key_material[:32])

        def anonymize_location(self, lat: float, lng: float, precision: str) -> dict:
            """Reduce precision for privacy"""
            precision_map = {
                'exact': 7,      # 150m
                'approximate': 5,  # 5km
                'city': 4,       # 20km
                'region': 3      # 150km
            }

            geohash = geohash2.encode(lat, lng, precision=precision_map[precision])
            center = geohash2.decode(geohash)

            return {
                'lat': center[0],
                'lng': center[1],
                'precision': precision
            }
    ```

    ---

    ## Cost Optimization

    **Monthly cost for 50M users:**

    | Component | Configuration | Cost |
    |-----------|--------------|------|
    | **EC2 (API servers)** | 100 × c5.2xlarge | $20,000 |
    | **Kafka cluster** | 50 × r5.large | $10,000 |
    | **InfluxDB cluster** | 50 × r5.xlarge | $25,000 |
    | **Cassandra cluster** | 30 × i3.xlarge | $18,000 |
    | **Redis cluster** | 20 × r5.large | $8,000 |
    | **PostgreSQL** | 5 × db.r5.xlarge | $5,000 |
    | **S3 storage** | 1 PB | $20,000 |
    | **Data transfer** | 100 TB/month | $9,000 |
    | **Total** | | **$115,000/month** |

    **Cost per user:** $2.30/year

    **Optimization opportunities:**
    - Spot instances for processing: 70% savings
    - Reserved instances: 40% savings
    - Data compression: 60% storage savings
    - **Optimized total: $65,000/month**

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Time-series database** - InfluxDB for write-optimized location storage
    2. **Redis Geospatial** - Sub-100ms proximity queries with GEORADIUS
    3. **Kafka for ingestion** - Handle 10M+ updates/sec with replay capability
    4. **Kalman filtering** - Smooth GPS noise and reject outliers
    5. **Adaptive intervals** - Reduce battery drain from 20% to 3-5% per hour
    6. **Tiered storage** - Hot/warm/cold optimization saves 75% on storage costs
    7. **Grid-based geofencing** - 5000x speedup vs checking all geofences

    ---

    ## Interview Tips

    ✅ **Start with scale** - Emphasize write-heavy workload (3.8M updates/sec)

    ✅ **Discuss geospatial** - Redis Geohash, QuadTree for dense areas

    ✅ **Battery optimization** - Adaptive intervals, motion detection, WiFi/cell fallback

    ✅ **Geofencing** - Grid indexing, state machine for enter/exit/dwell

    ✅ **Privacy** - Encryption, data retention, GDPR compliance

    ✅ **Storage optimization** - Time-series DB, compression, tiered archival

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle GPS jumps?"** | Kalman filter + outlier detection (reject >1km jumps), road snapping for driving |
    | **"What if InfluxDB is slow?"** | Batch writes (1000x reduction), sharding by user_id, read replicas |
    | **"How to minimize battery drain?"** | Adaptive intervals (60s stationary, 4s driving), WiFi/cell fallback, geofence-based wake |
    | **"How to ensure geofence accuracy?"** | Grid indexing for fast lookup, state machine to prevent duplicate events, accuracy thresholds |
    | **"How to handle 10M updates/sec?"** | Kafka partitioning (1000 partitions), parallel consumers, async processing |
    | **"What about privacy?"** | Encrypt coordinates, anonymize in sharing, data retention policies, GDPR rights |
    | **"How to query location history fast?"** | Continuous aggregation (1-min intervals), indexed by user_id + timestamp, Cassandra for range queries |

    ---

    ## Real-World Examples

    **Uber/Lyft:**
    - Update every 4 seconds while driving
    - Geohash precision 6 (1km cells)
    - Batch writes to reduce database load
    - Adaptive intervals based on ride state

    **Find My (Apple):**
    - Bluetooth mesh network for offline tracking
    - End-to-end encryption
    - Crowdsourced location network
    - Privacy-first design (anonymized locations)

    **Life360:**
    - Family location sharing
    - Geofence notifications (arrive/leave)
    - Location history trails
    - Battery optimization modes

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Uber, DoorDash, Find My (Apple), Google Maps, Life360

---

*Master this problem and you'll be ready for: Lyft, Instacart, Postmates, Waze, Glympse*
