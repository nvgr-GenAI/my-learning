# Design Uber (Ride-Sharing Platform)

A ride-sharing platform that connects riders with drivers in real-time, handles location tracking, matching, and payment processing at global scale.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M+ riders, 5M drivers, 15M rides/day globally |
| **Key Challenges** | Real-time location tracking, efficient matching, pricing surge, ETA calculation |
| **Core Concepts** | Geospatial indexing (quadtree), WebSocket, supply-demand matching, payment processing |
| **Companies** | Uber, Lyft, Didi, Grab, Ola, DoorDash, Instacart |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Request Ride** | Rider requests ride with pickup/dropoff location | P0 (Must have) |
    | **Driver Matching** | Match rider with nearby available driver | P0 (Must have) |
    | **Real-time Tracking** | Track driver location until ride completion | P0 (Must have) |
    | **ETA Calculation** | Estimate arrival and trip time | P0 (Must have) |
    | **Pricing** | Calculate fare with surge pricing | P0 (Must have) |
    | **Payment** | Process payment after ride | P0 (Must have) |
    | **Rating** | Rider and driver rate each other | P1 (Should have) |
    | **Ride History** | View past rides and receipts | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Food delivery (Uber Eats)
    - Ride scheduling (advance booking)
    - Carpool/shared rides
    - Driver background checks
    - Fraud detection

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Ride-sharing is mission-critical for users |
    | **Latency (Matching)** | < 5s to find driver | Fast matching improves user experience |
    | **Latency (Tracking)** | < 1s location updates | Real-time tracking is critical |
    | **Accuracy** | < 10m GPS accuracy | Precise pickup/dropoff locations |
    | **Scalability** | Millions of concurrent rides | Handle peak hours (Friday nights) |
    | **Consistency** | Strong consistency for payments | No double charges or lost payments |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total riders: 100M monthly active
    Total drivers: 5M active drivers
    Daily rides: 15M rides/day
    Average ride duration: 20 minutes

    Active rides (peak):
    - Peak multiplier: 3x average
    - Average concurrent: 15M rides / 72 rides per day = 208K concurrent
    - Peak concurrent: 208K × 3 = 625K concurrent rides

    Location updates:
    - Drivers send location every 4 seconds
    - 5M drivers × 0.25 updates/sec = 1.25M updates/sec
    - During rides only: 625K drivers × 0.25 = 156K updates/sec

    Matching requests:
    - 15M rides/day = 173 requests/sec average
    - Peak: 520 requests/sec
    ```

    ### Storage Estimates

    ```
    Ride data:
    - Per ride: 2 KB (rider_id, driver_id, locations, price, time)
    - Daily: 15M × 2 KB = 30 GB/day
    - Yearly: 30 GB × 365 = 10.95 TB/year
    - 5 years: 55 TB

    Location history:
    - Per location update: 50 bytes (driver_id, lat, lng, timestamp)
    - Daily: 1.25M updates/sec × 86,400 = 108B updates/day
    - Storage: 108B × 50 bytes = 5.4 TB/day
    - With retention (30 days): 162 TB

    User data:
    - 100M riders × 5 KB = 500 GB
    - 5M drivers × 10 KB = 50 GB
    - Total: 550 GB

    Total: 55 TB (rides) + 162 TB (locations) + 0.55 TB (users) ≈ 217 TB
    ```

    ### Memory Estimates (Caching)

    ```
    Active driver locations:
    - 5M drivers × 100 bytes = 500 MB

    Active ride state:
    - 625K concurrent rides × 1 KB = 625 MB

    Hot geohash lookups:
    - 100K cells × 10 KB (list of drivers) = 1 GB

    User sessions:
    - 10M concurrent users × 10 KB = 100 GB

    Total cache: 102 GB (mostly user sessions)
    ```

    ---

    ## Key Assumptions

    1. Average ride duration: 20 minutes
    2. Driver location updates: every 4 seconds
    3. GPS accuracy: 10 meters
    4. Peak to average ratio: 3:1 (Friday night vs Tuesday afternoon)
    5. 70% of rides completed successfully (30% cancelled)
    6. Average fare: $15 (varies by city)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Geospatial indexing** - Efficiently find nearby drivers
    2. **Real-time communication** - WebSocket for location updates
    3. **Event-driven architecture** - Kafka for ride lifecycle events
    4. **Microservices** - Separate services for matching, routing, pricing
    5. **Strong consistency for payments** - ACID transactions

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Rider_App[Rider Mobile App]
            Driver_App[Driver Mobile App]
        end

        subgraph "API Gateway"
            LB[Load Balancer]
            API_GW[API Gateway<br/>Auth, Rate limiting]
        end

        subgraph "Core Services"
            Matching[Matching Service<br/>Find nearby drivers]
            Tracking[Tracking Service<br/>Location updates]
            Routing[Routing Service<br/>ETA, route calculation]
            Pricing[Pricing Service<br/>Fare calculation]
            Payment[Payment Service<br/>Process payments]
            Notification[Notification Service<br/>Push notifications]
        end

        subgraph "Real-time Layer"
            WebSocket[WebSocket Server<br/>Location streaming]
        end

        subgraph "Geospatial"
            Geo_Index[Geospatial Index<br/>Redis Geohash/QuadTree]
        end

        subgraph "Caching"
            Redis_Location[Redis<br/>Driver locations]
            Redis_Ride[Redis<br/>Active ride state]
        end

        subgraph "Storage"
            Ride_DB[(Ride DB<br/>PostgreSQL<br/>ACID for payments)]
            Location_DB[(Location DB<br/>Cassandra<br/>Time-series)]
            User_DB[(User DB<br/>PostgreSQL)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Ride events]
        end

        subgraph "External Services"
            Maps_API[Google Maps API<br/>Routing, ETA]
            Payment_Gateway[Stripe/Braintree<br/>Payment processing]
        end

        Rider_App --> LB
        Driver_App --> LB

        LB --> API_GW
        API_GW --> Matching
        API_GW --> Tracking
        API_GW --> Routing
        API_GW --> Pricing
        API_GW --> Payment

        Driver_App --> WebSocket
        WebSocket --> Tracking
        Tracking --> Geo_Index
        Tracking --> Redis_Location
        Tracking --> Location_DB

        Matching --> Geo_Index
        Matching --> Redis_Location
        Matching --> Kafka

        Routing --> Maps_API
        Pricing --> Maps_API

        Payment --> Payment_Gateway
        Payment --> Ride_DB

        Kafka --> Notification
        Notification --> Rider_App
        Notification --> Driver_App

        Matching --> Ride_DB
        Tracking --> Ride_DB

        style LB fill:#e1f5ff
        style Geo_Index fill:#e8f5e9
        style Redis_Location fill:#fff4e1
        style Redis_Ride fill:#fff4e1
        style Ride_DB fill:#ffe1e1
        style Location_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis Geohash** | Find nearby drivers in < 10ms | Database spatial queries (too slow), in-memory scan (doesn't scale) |
    | **WebSocket** | Real-time bidirectional location updates | Polling (wasteful, high latency), Server-Sent Events (one-way only) |
    | **Kafka** | Reliable event streaming for ride lifecycle | Direct API calls (no retry, tight coupling), RabbitMQ (lower throughput) |
    | **PostgreSQL (Rides)** | ACID guarantees for payment transactions | NoSQL (no transactions, eventual consistency unacceptable for money) |
    | **Cassandra (Locations)** | High write throughput for location updates (1.25M/sec) | PostgreSQL (can't handle write volume) |
    | **Google Maps API** | Accurate routing and ETA | Build custom routing (reinventing wheel, less accurate) |

    **Key Trade-off:** We chose **strong consistency for payments** but **eventual consistency for location tracking**. Missing a location update is acceptable, but double-charging is not.

    ---

    ## API Design

    ### 1. Request Ride

    **Request:**
    ```http
    POST /api/v1/rides
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "pickup": {
        "lat": 37.7749,
        "lng": -122.4194,
        "address": "123 Market St, San Francisco"
      },
      "dropoff": {
        "lat": 37.8044,
        "lng": -122.2712,
        "address": "Oakland Airport"
      },
      "ride_type": "uberX",  // uberX, uberXL, uberBlack
      "passengers": 1
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "ride_id": "ride_abc123",
      "status": "searching",
      "estimated_price": {
        "amount": 25.50,
        "currency": "USD",
        "surge_multiplier": 1.2
      },
      "estimated_arrival": 300,  // seconds
      "estimated_duration": 1200  // seconds
    }
    ```

    **Design Notes:**

    - Return immediately with estimated price and ETA
    - Actual driver matching happens asynchronously
    - WebSocket connection established for real-time updates

    ---

    ### 2. Update Driver Location

    **Request (via WebSocket):**
    ```json
    {
      "type": "location_update",
      "driver_id": "driver_xyz789",
      "location": {
        "lat": 37.7750,
        "lng": -122.4195
      },
      "heading": 45,  // degrees
      "speed": 15.5,  // m/s
      "timestamp": 1643712000000
    }
    ```

    **Response:**
    ```json
    {
      "status": "acknowledged",
      "next_update": 4000  // milliseconds
    }
    ```

    **Design Notes:**

    - WebSocket for continuous updates (every 4 seconds)
    - Batch updates to reduce overhead
    - Server validates timestamp to detect stale data

    ---

    ### 3. Get Nearby Drivers (Internal API)

    **Request:**
    ```http
    GET /internal/api/v1/drivers/nearby?lat=37.7749&lng=-122.4194&radius=5000&ride_type=uberX&limit=20
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "drivers": [
        {
          "driver_id": "driver_xyz789",
          "location": {
            "lat": 37.7755,
            "lng": -122.4200
          },
          "rating": 4.9,
          "vehicle_type": "uberX",
          "eta_seconds": 180
        },
        // ... 19 more drivers
      ],
      "query_time_ms": 8
    }
    ```

    ---

    ## Database Schema

    ### Rides (PostgreSQL)

    ```sql
    -- Rides table (ACID transactions critical)
    CREATE TABLE rides (
        ride_id UUID PRIMARY KEY,
        rider_id UUID NOT NULL,
        driver_id UUID,
        status VARCHAR(20) NOT NULL,  -- requested, accepted, arrived, started, completed, cancelled
        pickup_location POINT NOT NULL,
        dropoff_location POINT NOT NULL,
        pickup_address TEXT,
        dropoff_address TEXT,
        ride_type VARCHAR(20),
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accepted_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        estimated_price DECIMAL(10, 2),
        actual_price DECIMAL(10, 2),
        surge_multiplier DECIMAL(3, 2) DEFAULT 1.0,
        distance_km DECIMAL(8, 2),
        duration_seconds INT,
        payment_status VARCHAR(20),  -- pending, completed, failed
        INDEX idx_rider (rider_id, requested_at),
        INDEX idx_driver (driver_id, requested_at),
        INDEX idx_status (status)
    );

    -- Ensure ACID: prevent double-booking drivers
    CREATE UNIQUE INDEX idx_driver_active ON rides (driver_id)
    WHERE status IN ('accepted', 'arrived', 'started');
    ```

    ---

    ### Driver Locations (Cassandra)

    ```sql
    -- Time-series location data
    CREATE TABLE driver_locations (
        driver_id UUID,
        timestamp TIMESTAMP,
        latitude DOUBLE,
        longitude DOUBLE,
        heading INT,
        speed DOUBLE,
        PRIMARY KEY (driver_id, timestamp)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);

    -- Current location (frequently updated)
    CREATE TABLE driver_current_location (
        driver_id UUID PRIMARY KEY,
        latitude DOUBLE,
        longitude DOUBLE,
        heading INT,
        speed DOUBLE,
        last_update TIMESTAMP,
        status VARCHAR(20)  -- available, on_ride, offline
    );
    ```

    **Why separate tables:**
    - `driver_current_location`: Latest position for matching (hot data)
    - `driver_locations`: Historical track for analytics (cold data)

    ---

    ## Data Flow Diagrams

    ### Ride Request Flow

    ```mermaid
    sequenceDiagram
        participant Rider
        participant API
        participant Matching
        participant Geo_Index
        participant Driver
        participant Kafka
        participant Notification

        Rider->>API: POST /api/v1/rides (pickup, dropoff)
        API->>API: Validate, calculate ETA & price
        API->>DB: INSERT ride (status: requested)
        API-->>Rider: ride_id, estimated_price

        API->>Matching: Find nearby drivers
        Matching->>Geo_Index: Get drivers within 5km radius
        Geo_Index-->>Matching: [driver1, driver2, ...]

        Matching->>Matching: Rank drivers (ETA, rating)
        Matching->>Driver: Send ride request (top 3 drivers)

        alt Driver accepts
            Driver-->>Matching: Accept ride
            Matching->>DB: UPDATE ride (status: accepted, driver_id)
            Matching->>Kafka: Publish ride_accepted event
            Kafka->>Notification: Notify rider
            Notification->>Rider: Driver assigned, ETA
        else No driver accepts (30s timeout)
            Matching->>Matching: Send to next batch of drivers
        end
    ```

    **Flow Explanation:**

    1. **Rider requests** - Submit pickup/dropoff, get ride_id
    2. **Find nearby drivers** - Query geospatial index (< 10ms)
    3. **Rank drivers** - By ETA, rating, acceptance rate
    4. **Send requests** - Push notification to top 3 drivers
    5. **First accepts** - Assign driver, notify rider
    6. **Timeout fallback** - Try next batch if no acceptance

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Uber subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Geospatial Indexing** | How to find nearby drivers in < 10ms? | Redis Geohash + QuadTree |
    | **Driver Matching** | How to match rider with best driver? | Multi-criteria optimization + timeout fallback |
    | **Real-time Tracking** | How to track 5M drivers in real-time? | WebSocket + location aggregation |
    | **Surge Pricing** | How to calculate dynamic pricing? | Supply-demand ratio + historical patterns |

    ---

    === "🗺️ Geospatial Indexing"

        ## The Challenge

        **Problem:** Find drivers within 5km of rider in < 10ms from 5M driver locations.

        **Naive approach:** Check distance to all 5M drivers. **Too slow:** 5M distance calculations = seconds.

        **Solution:** Geospatial indexing (Geohash + Redis)

        ---

        ## Geohash Explained

        **Concept:** Encode (lat, lng) as short string. Nearby locations have similar prefixes.

        **How it works:**

        1. **Encode location:** (37.7749, -122.4194) → "9q8yy"
        2. **Store in Redis:** Hash map `geohash:9q8yy` → [driver1, driver2, ...]
        3. **Search nearby:** Get all geohashes with prefix "9q8"

        **Geohash precision:**

        | Length | Cell Size | Use Case |
        |--------|-----------|----------|
        | 4 | 20 km × 20 km | City-level |
        | 5 | 5 km × 5 km | **Ride matching** |
        | 6 | 1 km × 1 km | Precise location |
        | 7 | 150 m × 150 m | Building-level |

        **Implementation:**

        ```python
        import geohash2
        import redis

        class GeospatialIndex:
            """Geospatial indexing for driver locations"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.precision = 6  # ~1 km precision

            def update_driver_location(self, driver_id: str, lat: float, lng: float):
                """
                Update driver location in geospatial index

                Args:
                    driver_id: Driver identifier
                    lat: Latitude
                    lng: Longitude
                """
                # Encode to geohash
                ghash = geohash2.encode(lat, lng, precision=self.precision)

                # Remove from old geohash (if exists)
                old_ghash = self.redis.hget(f"driver:{driver_id}", "geohash")
                if old_ghash and old_ghash != ghash:
                    self.redis.srem(f"geohash:{old_ghash}", driver_id)

                # Add to new geohash
                self.redis.sadd(f"geohash:{ghash}", driver_id)

                # Store current location
                self.redis.hset(f"driver:{driver_id}", mapping={
                    "lat": lat,
                    "lng": lng,
                    "geohash": ghash,
                    "last_update": time.time()
                })

            def find_nearby_drivers(
                self,
                lat: float,
                lng: float,
                radius_km: float = 5,
                limit: int = 20
            ) -> List[dict]:
                """
                Find drivers within radius

                Returns:
                    List of {driver_id, lat, lng, distance}
                """
                # Get geohash for location
                center_ghash = geohash2.encode(lat, lng, precision=self.precision)

                # Get all neighboring geohashes (9 cells: center + 8 neighbors)
                search_cells = self._get_neighbors(center_ghash)

                # Collect drivers from all cells
                candidate_drivers = set()
                for cell in search_cells:
                    drivers = self.redis.smembers(f"geohash:{cell}")
                    candidate_drivers.update(drivers)

                # Calculate exact distance and filter
                nearby_drivers = []
                for driver_id in candidate_drivers:
                    driver_data = self.redis.hgetall(f"driver:{driver_id}")
                    if not driver_data:
                        continue

                    driver_lat = float(driver_data['lat'])
                    driver_lng = float(driver_data['lng'])

                    # Haversine distance
                    distance = self._calculate_distance(lat, lng, driver_lat, driver_lng)

                    if distance <= radius_km:
                        nearby_drivers.append({
                            'driver_id': driver_id,
                            'lat': driver_lat,
                            'lng': driver_lng,
                            'distance_km': distance
                        })

                # Sort by distance and limit
                nearby_drivers.sort(key=lambda d: d['distance_km'])
                return nearby_drivers[:limit]

            def _get_neighbors(self, geohash: str) -> List[str]:
                """Get geohash and all 8 neighbors"""
                neighbors = geohash2.neighbors(geohash)
                return [geohash] + list(neighbors.values())

            def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """Haversine distance in kilometers"""
                from math import radians, cos, sin, asin, sqrt

                lon1, lat1, lon2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                km = 6371 * c
                return km
        ```

        **Performance:**

        - Geohash lookup: O(1) - Redis hash lookup
        - Neighbor scan: O(9) - Check 9 cells
        - Distance calculation: O(N) where N = candidates (~50-100 drivers)
        - **Total: < 10ms** for typical query

        ---

        ## Alternative: QuadTree

        **When Geohash isn't enough:** Very dense areas (downtown Manhattan) may have 1000+ drivers in single geohash cell.

        **Solution:** QuadTree for hierarchical spatial partitioning.

        **Concept:** Recursively divide space into 4 quadrants until each has < 100 drivers.

        **Trade-offs:**

        | Approach | Pros | Cons | Use Case |
        |----------|------|------|----------|
        | **Geohash** | Simple, Redis-native, fast | Poor for very dense areas | **Most cities** |
        | **QuadTree** | Handles density better, adaptive | More complex, custom implementation | Very dense cities (NYC, Tokyo) |

    === "🎯 Driver Matching Algorithm"

        ## The Challenge

        **Problem:** Match rider with best driver among 20 nearby candidates.

        **Criteria:**

        1. **ETA:** Closest driver (minimize wait time)
        2. **Rating:** Higher-rated drivers preferred
        3. **Acceptance rate:** Drivers who accept more rides
        4. **Ride direction:** Driver already heading toward pickup

        ---

        ## Matching Algorithm

        **Multi-criteria scoring:**

        ```python
        class DriverMatcher:
            """Match rider with optimal driver"""

            WEIGHTS = {
                'eta': 0.5,           # 50% weight on ETA
                'rating': 0.2,        # 20% on rating
                'acceptance_rate': 0.2,  # 20% on acceptance
                'direction': 0.1       # 10% on direction
            }

            def match_driver(self, ride_request: dict) -> str:
                """
                Find best driver for ride request

                Args:
                    ride_request: {pickup_lat, pickup_lng, ride_type}

                Returns:
                    driver_id of best match
                """
                pickup_lat = ride_request['pickup_lat']
                pickup_lng = ride_request['pickup_lng']

                # Get nearby available drivers
                candidates = geo_index.find_nearby_drivers(
                    pickup_lat,
                    pickup_lng,
                    radius_km=5,
                    limit=50
                )

                # Filter by ride type
                ride_type = ride_request['ride_type']
                candidates = [d for d in candidates if d['vehicle_type'] == ride_type]

                if not candidates:
                    return None  # No available drivers

                # Score each candidate
                scored_drivers = []
                for driver in candidates:
                    score = self._calculate_score(driver, pickup_lat, pickup_lng)
                    scored_drivers.append((driver['driver_id'], score))

                # Sort by score (descending)
                scored_drivers.sort(key=lambda x: x[1], reverse=True)

                # Send request to top 3 drivers simultaneously
                top_drivers = [d[0] for d in scored_drivers[:3]]
                accepted_driver = self._send_ride_requests(top_drivers, ride_request)

                # If no acceptance, try next batch
                if not accepted_driver and len(scored_drivers) > 3:
                    next_batch = [d[0] for d in scored_drivers[3:6]]
                    accepted_driver = self._send_ride_requests(next_batch, ride_request)

                return accepted_driver

            def _calculate_score(self, driver: dict, pickup_lat: float, pickup_lng: float) -> float:
                """
                Calculate driver score (0-1, higher is better)

                Factors:
                - ETA (distance to pickup)
                - Driver rating
                - Acceptance rate
                - Direction alignment
                """
                score = 0

                # 1. ETA score (inverse distance, normalized)
                distance_km = driver['distance_km']
                eta_score = max(0, 1 - (distance_km / 10))  # Normalize to 10km max
                score += self.WEIGHTS['eta'] * eta_score

                # 2. Rating score (0-5 scale, normalize to 0-1)
                rating = driver.get('rating', 4.5)
                rating_score = (rating - 3) / 2  # 3.0=0, 5.0=1
                score += self.WEIGHTS['rating'] * rating_score

                # 3. Acceptance rate score
                acceptance_rate = driver.get('acceptance_rate', 0.8)
                score += self.WEIGHTS['acceptance_rate'] * acceptance_rate

                # 4. Direction score (if driver heading toward pickup)
                if 'heading' in driver:
                    direction_score = self._calculate_direction_alignment(
                        driver['lat'],
                        driver['lng'],
                        driver['heading'],
                        pickup_lat,
                        pickup_lng
                    )
                    score += self.WEIGHTS['direction'] * direction_score

                return score

            def _send_ride_requests(self, driver_ids: List[str], ride_request: dict) -> str:
                """
                Send ride request to multiple drivers, return first to accept

                Args:
                    driver_ids: List of driver IDs
                    ride_request: Ride details

                Returns:
                    driver_id of accepting driver, or None
                """
                # Send push notifications to all drivers simultaneously
                for driver_id in driver_ids:
                    notification_service.send_ride_request(driver_id, ride_request)

                # Wait for first acceptance (30 second timeout)
                timeout = 30
                start_time = time.time()

                while time.time() - start_time < timeout:
                    # Check for acceptance (poll Redis)
                    accepted = redis.get(f"ride_request:{ride_request['ride_id']}:accepted")
                    if accepted:
                        accepting_driver = accepted.decode()

                        # Cancel requests to other drivers
                        for driver_id in driver_ids:
                            if driver_id != accepting_driver:
                                notification_service.cancel_ride_request(driver_id, ride_request['ride_id'])

                        return accepting_driver

                    time.sleep(0.5)  # Poll every 500ms

                # Timeout - no acceptance
                return None
        ```

        ---

        ## Handling Edge Cases

        **1. No available drivers:**
        - Widen search radius (5km → 10km → 20km)
        - Notify rider of expected wait time
        - Suggest alternative ride types

        **2. Driver rejects:**
        - Track rejection reasons (too far, wrong direction)
        - Lower driver's score temporarily
        - Move to next candidates

        **3. Simultaneous requests:**
        - Use Redis transaction to prevent double-booking
        - First request to complete wins
        - Release other requests immediately

    === "📍 Real-time Tracking"

        ## The Challenge

        **Problem:** Track 5M driver locations in real-time (1.25M updates/sec).

        **Requirements:**

        - Low latency: < 1s from driver to rider app
        - High throughput: 1.25M updates/sec
        - Scalable: 10M drivers in future

        ---

        ## WebSocket Architecture

        **Connection management:**

        ```
        5M drivers
        ├── 50 WebSocket servers (100K connections each)
        │   ├── WS-1: drivers 0-100K
        │   ├── WS-2: drivers 100K-200K
        │   └── ...
        └── Redis Pub/Sub (routing layer)
        ```

        **Implementation:**

        ```python
        import asyncio
        import websockets

        class LocationTracker:
            """Real-time driver location tracking"""

            def __init__(self):
                self.connections = {}  # driver_id -> WebSocket
                self.redis = redis.asyncio.Redis()

            async def handle_driver_connection(self, websocket, driver_id: str):
                """Handle WebSocket connection from driver"""
                self.connections[driver_id] = websocket
                logger.info(f"Driver {driver_id} connected")

                try:
                    async for message in websocket:
                        await self.process_location_update(driver_id, message)
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Driver {driver_id} disconnected")
                finally:
                    del self.connections[driver_id]

            async def process_location_update(self, driver_id: str, message: str):
                """
                Process incoming location update

                Message format: {"lat": 37.7749, "lng": -122.4194, "heading": 45}
                """
                data = json.loads(message)
                lat = data['lat']
                lng = data['lng']

                # Update geospatial index
                geo_index.update_driver_location(driver_id, lat, lng)

                # Store in time-series database (async)
                asyncio.create_task(
                    self.store_location_history(driver_id, lat, lng, data.get('heading'), data.get('speed'))
                )

                # Check if driver is on active ride
                ride_id = await self.redis.get(f"driver:{driver_id}:active_ride")
                if ride_id:
                    # Notify rider of driver location
                    await self.notify_rider(ride_id, lat, lng)

            async def notify_rider(self, ride_id: str, driver_lat: float, driver_lng: float):
                """Send driver location to rider via WebSocket"""
                rider_id = await self.redis.hget(f"ride:{ride_id}", "rider_id")

                # Publish to Redis (rider's WebSocket server will receive)
                await self.redis.publish(
                    f"rider:{rider_id}:location",
                    json.dumps({"lat": driver_lat, "lng": driver_lng})
                )
        ```

        ---

        ## Location Aggregation

        **Problem:** 1.25M updates/sec overwhelming database.

        **Solution:** Aggregate updates before writing to database.

        ```python
        class LocationAggregator:
            """Batch location updates for efficient database writes"""

            def __init__(self):
                self.buffer = defaultdict(list)
                self.batch_size = 1000
                self.flush_interval = 10  # seconds

            async def add_location(self, driver_id: str, location: dict):
                """Add location to buffer"""
                self.buffer[driver_id].append(location)

                # Flush if batch full
                if len(self.buffer[driver_id]) >= 100:
                    await self.flush_driver(driver_id)

            async def flush_driver(self, driver_id: str):
                """Write buffered locations to Cassandra"""
                locations = self.buffer[driver_id]
                if not locations:
                    return

                # Batch insert
                await cassandra.execute_batch([
                    {
                        'driver_id': driver_id,
                        'timestamp': loc['timestamp'],
                        'latitude': loc['lat'],
                        'longitude': loc['lng']
                    }
                    for loc in locations
                ])

                self.buffer[driver_id] = []
        ```

        **Benefits:**

        - Reduce database writes from 1.25M/sec to 12.5K/sec (100x reduction)
        - Lower database load and cost
        - Acceptable delay (< 10s for historical data)

    === "💰 Surge Pricing Algorithm"

        ## The Challenge

        **Problem:** Balance supply (drivers) and demand (riders) dynamically.

        **Goal:** Incentivize more drivers during high demand (Friday night, rain, events).

        ---

        ## Surge Calculation

        **Formula:**

        ```
        surge_multiplier = 1.0 + (demand_ratio - 1.0) × sensitivity

        where:
        demand_ratio = pending_rides / available_drivers
        sensitivity = 0.5 (tune based on market)
        ```

        **Implementation:**

        ```python
        class SurgePricing:
            """Calculate dynamic surge pricing"""

            BASE_SENSITIVITY = 0.5
            MIN_SURGE = 1.0
            MAX_SURGE = 3.0

            def calculate_surge(self, lat: float, lng: float, radius_km: float = 2) -> float:
                """
                Calculate surge multiplier for area

                Args:
                    lat, lng: Center of area
                    radius_km: Search radius

                Returns:
                    Surge multiplier (1.0 - 3.0)
                """
                # Get supply (available drivers)
                available_drivers = geo_index.find_nearby_drivers(
                    lat, lng, radius_km, limit=1000
                )
                supply = len([d for d in available_drivers if d['status'] == 'available'])

                # Get demand (pending ride requests)
                pending_rides = db.query("""
                    SELECT COUNT(*) FROM rides
                    WHERE status = 'requested'
                      AND ST_DWithin(pickup_location, ST_MakePoint(%s, %s)::geography, %s)
                      AND requested_at > NOW() - INTERVAL '5 minutes'
                """, (lng, lat, radius_km * 1000))['count']

                if supply == 0:
                    return self.MAX_SURGE  # No drivers, max surge

                # Calculate demand ratio
                demand_ratio = pending_rides / supply

                # Apply surge formula
                surge = 1.0 + max(0, (demand_ratio - 1.0)) * self.BASE_SENSITIVITY

                # Clamp to min/max
                surge = max(self.MIN_SURGE, min(surge, self.MAX_SURGE))

                # Apply historical adjustment (learn from past patterns)
                surge = self._apply_historical_adjustment(surge, lat, lng)

                return round(surge, 1)

            def _apply_historical_adjustment(self, surge: float, lat: float, lng: float) -> float:
                """
                Adjust surge based on historical patterns

                Example: Always high demand at airport 6-8pm
                """
                hour = datetime.utcnow().hour
                day_of_week = datetime.utcnow().weekday()

                # Get historical average surge for this location/time
                historical_surge = cache.get(f"historical_surge:{lat:.2f}:{lng:.2f}:{hour}:{day_of_week}")

                if historical_surge:
                    # Blend current and historical (70% current, 30% historical)
                    surge = 0.7 * surge + 0.3 * float(historical_surge)

                return surge
        ```

        ---

        ## Surge Transparency

        **User experience:**

        1. **Show surge upfront:** "Fares are higher due to increased demand"
        2. **Explain why:** "Raining + Friday night"
        3. **Show map:** Heat map of surge areas
        4. **Allow waiting:** "Surge usually drops in 15-20 minutes"

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Uber from 1M to 100M riders.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Geospatial queries** | ✅ Yes | Redis Geohash sharding, 50 nodes |
    | **WebSocket connections** | ✅ Yes | 50 servers (100K connections each) |
    | **Location writes** | ✅ Yes | Cassandra cluster (100 nodes), batch writes |
    | **Payment processing** | 🟡 Moderate | PostgreSQL with read replicas, async processing |

    ---

    ## Cost Optimization

    **Monthly cost at 100M riders:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API/WS)** | $216,000 (1,000 servers) |
    | **Redis (Geo)** | $54,000 (50 nodes) |
    | **Cassandra** | $162,000 (100 nodes) |
    | **PostgreSQL** | $43,200 (20 shards) |
    | **Maps API** | $1,500,000 (15M rides × $0.10/route) |
    | **Total** | **$2M/month** |

    **Revenue:** Uber takes 25% commission. Average ride $15. 15M rides/day = $225M/month revenue. Infrastructure is 0.9% of revenue.

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Redis Geohash** - Find nearby drivers in < 10ms
    2. **WebSocket** - Real-time location updates (1.25M/sec)
    3. **Multi-criteria matching** - ETA + rating + acceptance rate
    4. **Strong consistency for payments** - PostgreSQL ACID transactions
    5. **Surge pricing** - Supply-demand ratio + historical patterns
    6. **Location aggregation** - Batch writes (100x reduction)

    ---

    ## Interview Tips

    ✅ **Start with geospatial** - How to find nearby drivers efficiently

    ✅ **Discuss real-time tracking** - WebSocket architecture

    ✅ **Matching algorithm** - Multi-criteria scoring

    ✅ **Surge pricing** - Supply-demand balance

    ✅ **Consistency** - ACID for payments vs eventual for locations

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent driver double-booking?"** | PostgreSQL unique index on (driver_id) WHERE status IN ('accepted', 'started') |
    | **"What if WebSocket connection drops?"** | Fallback to HTTP long polling, reconnect with exponential backoff |
    | **"How to calculate ETA accurately?"** | Google Maps API with real-time traffic, historical patterns |
    | **"How to handle payment failures?"** | Retry with exponential backoff, email rider, allow cash payment |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Uber, Lyft, Didi, Grab, Ola, DoorDash

---

*Master this problem and you'll be ready for: Lyft, DoorDash, Instacart, Postmates*
