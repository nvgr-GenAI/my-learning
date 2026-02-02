# Design Google Maps

A navigation and mapping platform that provides real-time routing, turn-by-turn navigation, traffic information, and location-based services to billions of users worldwide.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1B users, 25M miles of roads, 1B route requests/day, 5B location updates/day |
| **Key Challenges** | Routing algorithms (Dijkstra/A*), real-time traffic prediction, graph storage, ETA accuracy, tile serving |
| **Core Concepts** | Graph databases, A* algorithm, traffic ML models, map tiles, geospatial indexing, contraction hierarchies |
| **Companies** | Google, Apple, TomTom, HERE Technologies, Waze, Uber, Lyft |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Route Calculation** | Find optimal route between two points | P0 (Must have) |
    | **Turn-by-turn Navigation** | Real-time voice guidance and directions | P0 (Must have) |
    | **Real-time Traffic** | Display current traffic conditions | P0 (Must have) |
    | **ETA Calculation** | Accurate arrival time prediction | P0 (Must have) |
    | **Map Display** | Render map tiles at various zoom levels | P0 (Must have) |
    | **Search Places** | Search for locations, businesses, addresses | P0 (Must have) |
    | **Alternative Routes** | Show 2-3 alternative route options | P1 (Should have) |
    | **Offline Maps** | Download maps for offline use | P1 (Should have) |
    | **Live Location Sharing** | Share real-time location with contacts | P2 (Nice to have) |
    | **Street View** | 360° panoramic street-level imagery | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Satellite imagery processing
    - 3D building rendering
    - Indoor mapping
    - Augmented reality navigation
    - User reviews and ratings system
    - Public transit scheduling

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Navigation is mission-critical for drivers |
    | **Latency (Route Calculation)** | < 100ms p95 | Users expect instant route results |
    | **Latency (Map Tiles)** | < 50ms p95 | Smooth panning/zooming experience |
    | **ETA Accuracy** | 90% within ±5 minutes | Trust is critical for user retention |
    | **Consistency** | Eventual consistency | Brief delays acceptable for traffic updates |
    | **Scalability** | Handle 1B users, 25M miles of roads | Global scale with billions of daily requests |
    | **Real-time** | Traffic updates within 30 seconds | Recent data critical for accurate routing |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 500M
    Monthly Active Users (MAU): 1B

    Route requests:
    - Routes per DAU: ~2 routes/day (commute, errands)
    - Daily routes: 500M × 2 = 1B routes/day
    - Route QPS: 1B / 86,400 = ~11,600 routes/sec
    - Peak QPS: 3x average = ~35,000 routes/sec (rush hour)

    Map tile requests:
    - Tiles per route view: ~50 tiles (various zoom levels)
    - Daily tiles: 1B routes × 50 = 50B tiles/day
    - Tile QPS: 50B / 86,400 = ~579,000 tiles/sec
    - Peak QPS: 3x average = ~1.7M tiles/sec

    Location updates (active navigation):
    - Active navigations: 50M concurrent (10% of DAU)
    - Update frequency: 2 updates/second
    - Location update QPS: 50M × 2 = 100M updates/sec

    Traffic data:
    - GPS probes: 100M active vehicles
    - Update frequency: Every 10 seconds
    - Traffic update QPS: 100M / 10 = 10M updates/sec

    Search queries:
    - Searches per DAU: ~3 searches/day
    - Daily searches: 500M × 3 = 1.5B searches/day
    - Search QPS: 1.5B / 86,400 = ~17,400 searches/sec

    Total Read QPS: ~579K (tiles) + 17.4K (search) = ~596K
    Total Write QPS: 100M (location) + 10M (traffic) = ~110M
    Read/Write ratio: 1:185 (extremely write-heavy)
    ```

    ### Storage Estimates

    ```
    Road network graph:
    - Total road segments: 100M segments (25M miles, avg 0.25 mile/segment)
    - Per segment: 200 bytes (coordinates, attributes, speed limits)
    - Graph storage: 100M × 200 bytes = 20 GB (raw)
    - With indexes: 20 GB × 3 = 60 GB

    Historical traffic data:
    - Traffic measurements per segment per hour: 1 record
    - 100M segments × 24 hours × 365 days × 2 years = 1.75 trillion records
    - Per record: 20 bytes (segment_id, timestamp, speed, volume)
    - Storage: 1.75T × 20 bytes = 35 TB (compressed: ~10 TB)

    Map tiles:
    - Zoom levels: 0-18 (19 levels)
    - Tiles at zoom 18: 68B tiles (256×256 pixels)
    - Average tile size: 10 KB (PNG compressed)
    - Total: 68B × 10 KB = 680 TB
    - With all zoom levels: ~850 TB

    Places database:
    - Total places: 200M businesses/POIs
    - Per place: 2 KB (name, address, coordinates, metadata)
    - Storage: 200M × 2 KB = 400 GB

    User data:
    - 1B users × 5 KB (preferences, history, saved places) = 5 TB

    Real-time traffic state:
    - Current traffic for 100M segments: 100M × 50 bytes = 5 GB (in-memory)

    Total: 60 GB (graph) + 10 TB (traffic history) + 850 TB (tiles) + 400 GB (places) + 5 TB (users) ≈ 866 TB
    ```

    ### Bandwidth Estimates

    ```
    Map tile egress:
    - 579,000 tiles/sec × 10 KB = 5.79 GB/sec ≈ 46 Gbps
    - Peak: 3x = 138 Gbps

    Location updates ingress:
    - 100M updates/sec × 50 bytes = 5 GB/sec ≈ 40 Gbps

    Route responses:
    - 11,600 routes/sec × 5 KB (route geometry) = 58 MB/sec ≈ 464 Mbps

    Total ingress: ~40 Gbps (location updates)
    Total egress: ~46 Gbps (CDN critical for tiles)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot tiles (major cities):
    - 1% of tiles (680M tiles) × 10 KB = 6.8 TB
    - Cache 10% hottest: 680 GB

    Road graph (in-memory):
    - Entire graph with indexes: 60 GB
    - Replicated across regions: 60 GB × 10 regions = 600 GB

    Current traffic state:
    - All segments: 5 GB
    - Replicated: 5 GB × 10 regions = 50 GB

    Route cache (recent routes):
    - 10M most common routes × 5 KB = 50 GB

    Places cache:
    - 10M hottest places × 2 KB = 20 GB

    Total cache: 680 GB + 600 GB + 50 GB + 50 GB + 20 GB ≈ 1.4 TB
    ```

    ---

    ## Key Assumptions

    1. Average trip distance: 10 miles, 20 minutes duration
    2. 10% of users are actively navigating at peak times
    3. Road network changes slowly (weekly updates acceptable)
    4. Traffic patterns are predictable based on historical data
    5. Tile requests heavily cacheable (90% cache hit rate)
    6. GPS probe data from smartphones provides sufficient coverage

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Graph-based routing:** Model road network as directed weighted graph
    2. **Hierarchical pathfinding:** Use contraction hierarchies for fast routing
    3. **ML-based ETA:** Predict traffic using historical patterns and real-time data
    4. **Tile-based rendering:** Pre-rendered tiles cached on CDN
    5. **Geospatial indexing:** Quad-tree/S2 geometry for spatial queries

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
            CarPlay[CarPlay/Android Auto]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Map tiles, static assets]
            LB[Load Balancer<br/>Geographic routing]
        end

        subgraph "API Layer"
            Route_API[Route Service<br/>Pathfinding, A*]
            Nav_API[Navigation Service<br/>Turn-by-turn guidance]
            Search_API[Search Service<br/>Places, geocoding]
            Tile_API[Tile Service<br/>Map rendering]
            Traffic_API[Traffic Service<br/>Real-time conditions]
        end

        subgraph "Processing Layer"
            Location_Processor[Location Processor<br/>GPS probe ingestion]
            Traffic_ML[Traffic Prediction<br/>ML models]
            ETA_Engine[ETA Engine<br/>Arrival time prediction]
            Route_Optimizer[Route Optimizer<br/>Batch pre-computation]
        end

        subgraph "Caching"
            Redis_Graph[Redis<br/>Road graph cache]
            Redis_Traffic[Redis<br/>Traffic state]
            Redis_Route[Redis<br/>Route cache]
            Redis_Tile[Redis<br/>Tile metadata]
        end

        subgraph "Storage"
            Graph_DB[(Graph DB<br/>Neo4j/JanusGraph<br/>Road network)]
            Traffic_DB[(TimeSeries DB<br/>InfluxDB<br/>Traffic history)]
            Places_DB[(PostgreSQL<br/>Sharded<br/>Places, POIs)]
            Tile_Storage[Object Storage<br/>S3<br/>Map tiles]
            User_DB[(PostgreSQL<br/>User preferences)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Location events]
        end

        subgraph "Geospatial"
            S2_Index[S2 Geometry<br/>Spatial indexing]
            QuadTree[QuadTree<br/>Tile indexing]
        end

        Mobile --> CDN
        Web --> CDN
        CarPlay --> LB
        Mobile --> LB
        Web --> LB

        CDN --> Tile_Storage

        LB --> Route_API
        LB --> Nav_API
        LB --> Search_API
        LB --> Tile_API
        LB --> Traffic_API

        Route_API --> Redis_Graph
        Route_API --> Redis_Route
        Route_API --> Graph_DB
        Route_API --> Traffic_API

        Nav_API --> Route_API
        Nav_API --> Kafka

        Search_API --> Places_DB
        Search_API --> S2_Index

        Tile_API --> Redis_Tile
        Tile_API --> Tile_Storage
        Tile_API --> QuadTree

        Traffic_API --> Redis_Traffic
        Traffic_API --> Traffic_DB

        Kafka --> Location_Processor
        Location_Processor --> Traffic_ML
        Traffic_ML --> ETA_Engine
        ETA_Engine --> Redis_Traffic
        ETA_Engine --> Traffic_DB

        Route_Optimizer --> Graph_DB
        Route_Optimizer --> Redis_Route

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Graph fill:#fff4e1
        style Redis_Traffic fill:#fff4e1
        style Redis_Route fill:#fff4e1
        style Redis_Tile fill:#fff4e1
        style Graph_DB fill:#e1f5e1
        style Traffic_DB fill:#ffe1e1
        style Places_DB fill:#ffe1e1
        style Tile_Storage fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Graph Database** | Optimized for pathfinding queries, native graph traversal | SQL with adjacency lists (too slow for routing), custom graph (complex) |
    | **Kafka** | Handle 110M location updates/sec, replay capability for ML training | Direct writes to DB (can't handle write volume), SQS (no ordering) |
    | **InfluxDB (TimeSeries)** | Optimized for time-series traffic data, compression, downsampling | PostgreSQL (slow for time-series), Cassandra (overkill for queries) |
    | **Redis (Graph Cache)** | Sub-millisecond graph queries, entire graph fits in memory | No cache (graph DB too slow for 35K QPS), Memcached (limited data structures) |
    | **S2 Geometry** | Hierarchical spatial indexing, covers earth with cells | PostGIS (slower for spatial queries), custom quad-tree (reinventing wheel) |
    | **Contraction Hierarchies** | Reduces routing time from O(E log V) to O(log V), millisecond routes | Pure Dijkstra (too slow), pre-compute all routes (infeasible storage) |

    **Key Trade-off:** We chose **speed over perfect accuracy**. Routes are optimized using cached graph with recent traffic, not real-time edge weights. 99% of users see optimal routes, 1% may see sub-optimal due to cache staleness.

    ---

    ## API Design

    ### 1. Calculate Route

    **Request:**
    ```http
    POST /api/v1/routes/calculate
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "origin": {
        "lat": 37.7749,
        "lng": -122.4194
      },
      "destination": {
        "lat": 37.3382,
        "lng": -121.8863
      },
      "departure_time": "2026-02-02T08:00:00Z",  // Optional (default: now)
      "mode": "driving",                          // driving, walking, bicycling, transit
      "alternatives": 3,                          // Number of alternative routes
      "avoid": ["tolls", "highways"],            // Optional restrictions
      "traffic_model": "best_guess"              // best_guess, optimistic, pessimistic
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "routes": [
        {
          "route_id": "route_abc123",
          "summary": "I-280 S",
          "distance_meters": 56000,
          "duration_seconds": 2400,
          "duration_in_traffic_seconds": 3000,
          "polyline": "encoded_polyline_string",  // Encoded route geometry
          "bounds": {
            "northeast": {"lat": 37.7749, "lng": -122.4194},
            "southwest": {"lat": 37.3382, "lng": -121.8863}
          },
          "legs": [
            {
              "distance_meters": 56000,
              "duration_seconds": 2400,
              "start_location": {"lat": 37.7749, "lng": -122.4194},
              "end_location": {"lat": 37.3382, "lng": -121.8863},
              "steps": [
                {
                  "distance_meters": 500,
                  "duration_seconds": 60,
                  "start_location": {"lat": 37.7749, "lng": -122.4194},
                  "end_location": {"lat": 37.7710, "lng": -122.4150},
                  "polyline": "encoded_step_polyline",
                  "instruction": "Head south on Van Ness Ave toward Market St",
                  "maneuver": "turn-right"
                }
                // ... more steps
              ]
            }
          ],
          "warnings": ["This route includes tolls"],
          "waypoint_order": [0, 1]  // For multi-waypoint routes
        }
        // ... alternative routes
      ],
      "geocoded_waypoints": [
        {
          "geocoder_status": "OK",
          "place_id": "ChIJIQBpAG2ahYAR_6128GcTUEo",
          "types": ["locality", "political"]
        }
      ],
      "computation_time_ms": 42
    }
    ```

    **Design Notes:**

    - Return immediately with best route (< 100ms)
    - Include traffic-adjusted ETA
    - Encode polyline using Google's encoding algorithm (compression)
    - Pre-compute common routes and cache

    ---

    ### 2. Get Real-time Traffic

    **Request:**
    ```http
    GET /api/v1/traffic?bounds=37.7,-122.5,37.8,-122.3&zoom=12
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "traffic_segments": [
        {
          "segment_id": "seg_12345",
          "polyline": "encoded_polyline",
          "speed_kmh": 25,
          "free_flow_speed_kmh": 65,
          "congestion_level": "heavy",  // free, light, moderate, heavy, stopped
          "timestamp": "2026-02-02T08:15:00Z"
        }
        // ... more segments
      ],
      "incidents": [
        {
          "incident_id": "inc_789",
          "type": "accident",
          "location": {"lat": 37.75, "lng": -122.42},
          "description": "Multi-vehicle accident blocking 2 lanes",
          "severity": "major",
          "start_time": "2026-02-02T07:30:00Z",
          "delay_minutes": 15
        }
      ],
      "timestamp": "2026-02-02T08:15:30Z",
      "expires_at": "2026-02-02T08:16:00Z"
    }
    ```

    **Design Notes:**

    - Update frequency: 30-60 seconds
    - Aggregate traffic by zoom level (fewer segments at lower zoom)
    - Include incidents from external sources (highway patrol, user reports)

    ---

    ### 3. Search Places

    **Request:**
    ```http
    GET /api/v1/places/search?query=coffee&location=37.7749,-122.4194&radius=5000
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "results": [
        {
          "place_id": "ChIJ...",
          "name": "Blue Bottle Coffee",
          "formatted_address": "66 Mint St, San Francisco, CA 94103",
          "geometry": {
            "location": {"lat": 37.7789, "lng": -122.4089},
            "viewport": {
              "northeast": {"lat": 37.7802, "lng": -122.4076},
              "southwest": {"lat": 37.7776, "lng": -122.4102}
            }
          },
          "types": ["cafe", "food", "point_of_interest"],
          "rating": 4.5,
          "user_ratings_total": 1250,
          "price_level": 2,
          "opening_hours": {
            "open_now": true
          },
          "distance_meters": 850
        }
        // ... more results
      ],
      "next_page_token": "token_xyz",
      "status": "OK"
    }
    ```

    **Design Notes:**

    - Use geospatial index (S2) for proximity search
    - Rank by relevance (distance, rating, popularity)
    - Cache popular queries

    ---

    ## Database Schema

    ### Road Network Graph (Neo4j)

    ```cypher
    // Node: Intersection/Junction
    CREATE (n:Node {
      node_id: "node_12345",
      lat: 37.7749,
      lng: -122.4194,
      elevation_m: 15
    })

    // Edge: Road Segment
    CREATE (n1:Node {node_id: "node_1"})-[r:ROAD_SEGMENT {
      segment_id: "seg_12345",
      length_meters: 500,
      road_name: "Market Street",
      road_type: "primary",
      speed_limit_kmh: 50,
      lanes: 4,
      one_way: false,
      turn_restrictions: ["no_left_turn"],
      // Pre-computed for routing
      avg_speed_kmh: 35,
      travel_time_sec: 51,
      // Contraction hierarchies
      ch_level: 5,  // 0 = lowest (local roads), 10 = highest (highways)
      ch_shortcuts: ["seg_67890", "seg_11111"]  // Pre-computed shortcuts
    }]->(n2:Node {node_id: "node_2"})

    // Queries
    // Find neighbors (used in Dijkstra/A*)
    MATCH (n:Node {node_id: "node_1"})-[r:ROAD_SEGMENT]->(neighbor:Node)
    RETURN neighbor, r

    // Find road by name
    MATCH (n1:Node)-[r:ROAD_SEGMENT]->(n2:Node)
    WHERE r.road_name = "Market Street"
    RETURN n1, r, n2
    ```

    **Why Graph Database:**

    - **Native graph traversal:** O(1) neighbor queries (critical for Dijkstra)
    - **Relationship properties:** Store road attributes on edges
    - **Pathfinding optimizations:** Built-in shortest path algorithms
    - **Schema flexibility:** Easy to add new road attributes

    ---

    ### Traffic Data (InfluxDB)

    ```sql
    -- Measurement: traffic_speed
    -- Tags: segment_id, road_type, day_of_week, hour
    -- Fields: speed_kmh, volume, occupancy
    -- Timestamp: observation time

    SELECT
      MEAN(speed_kmh) AS avg_speed,
      PERCENTILE(speed_kmh, 50) AS median_speed,
      MIN(speed_kmh) AS min_speed
    FROM traffic_speed
    WHERE
      segment_id = 'seg_12345'
      AND time > now() - 7d
    GROUP BY time(15m)

    -- Historical pattern query (for ML training)
    SELECT
      MEAN(speed_kmh) AS typical_speed
    FROM traffic_speed
    WHERE
      segment_id = 'seg_12345'
      AND day_of_week = 'Monday'
      AND hour = 8
      AND time > now() - 90d
    GROUP BY time(15m)
    ```

    ---

    ### Places (PostgreSQL + PostGIS)

    ```sql
    -- Places table (sharded by geohash)
    CREATE TABLE places (
        place_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        formatted_address TEXT,
        location GEOGRAPHY(POINT, 4326) NOT NULL,  -- PostGIS type
        types VARCHAR(50)[],
        rating DECIMAL(2, 1),
        user_ratings_total INT,
        price_level INT,
        phone_number VARCHAR(20),
        website TEXT,
        opening_hours JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) PARTITION BY HASH (place_id);

    -- Spatial index (critical for proximity queries)
    CREATE INDEX idx_places_location ON places USING GIST (location);

    -- Text search index
    CREATE INDEX idx_places_name ON places USING GIN (to_tsvector('english', name));

    -- Proximity search query
    SELECT
      place_id,
      name,
      ST_Distance(location, ST_MakePoint(-122.4194, 37.7749)::geography) AS distance_meters
    FROM places
    WHERE
      ST_DWithin(location, ST_MakePoint(-122.4194, 37.7749)::geography, 5000)  -- 5km radius
      AND types && ARRAY['cafe', 'restaurant']
    ORDER BY distance_meters
    LIMIT 20;
    ```

    ---

    ### Map Tiles (Metadata in PostgreSQL)

    ```sql
    -- Tile metadata (actual tiles in S3)
    CREATE TABLE tile_metadata (
        tile_id VARCHAR(50) PRIMARY KEY,  -- Format: z/x/y (e.g., "12/656/1582")
        zoom_level INT NOT NULL,
        tile_x INT NOT NULL,
        tile_y INT NOT NULL,
        bounds GEOGRAPHY(POLYGON, 4326),
        tile_url TEXT NOT NULL,  -- S3 URL
        version INT DEFAULT 1,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_tile_zoom_xy (zoom_level, tile_x, tile_y)
    );

    -- QuadTree for hierarchical tile indexing
    CREATE TABLE tile_quadtree (
        quadkey VARCHAR(20) PRIMARY KEY,  -- Bing Maps quadkey
        zoom_level INT NOT NULL,
        tile_id VARCHAR(50) NOT NULL REFERENCES tile_metadata(tile_id),
        parent_quadkey VARCHAR(19),  -- Parent tile
        INDEX idx_quadkey_zoom (quadkey, zoom_level)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Route Calculation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Route_API
        participant Redis_Cache
        participant Graph_DB
        participant Traffic_API
        participant ETA_Engine

        Client->>Route_API: POST /routes/calculate
        Route_API->>Route_API: Validate coordinates

        Route_API->>Redis_Cache: GET route_cache:origin:dest
        alt Cache HIT (20% of requests)
            Redis_Cache-->>Route_API: Cached route
            Route_API->>Traffic_API: Get current traffic
            Traffic_API-->>Route_API: Traffic data
            Route_API->>ETA_Engine: Recalculate ETA
            ETA_Engine-->>Route_API: Updated ETA
            Route_API-->>Client: 200 OK (route + ETA)
        else Cache MISS (80% of requests)
            Redis_Cache-->>Route_API: null

            Route_API->>Redis_Cache: GET graph:region
            alt Graph in cache
                Redis_Cache-->>Route_API: Graph data
            else Graph not cached
                Route_API->>Graph_DB: Query graph
                Graph_DB-->>Route_API: Graph data
                Route_API->>Redis_Cache: SET graph:region (TTL: 1h)
            end

            Route_API->>Traffic_API: Get current traffic for region
            Traffic_API-->>Route_API: Traffic speeds

            Route_API->>Route_API: Run A* with traffic weights
            Route_API->>Route_API: Find top 3 routes

            Route_API->>ETA_Engine: Calculate ETA for each route
            ETA_Engine-->>Route_API: ETAs

            Route_API->>Redis_Cache: SET route_cache (TTL: 5m)
            Route_API-->>Client: 200 OK (routes + ETAs)
        end
    ```

    **Flow Explanation:**

    1. **Check route cache** - Common routes cached for 5 minutes
    2. **Load graph** - Region graph cached in Redis (1 hour TTL)
    3. **Get traffic** - Current traffic speeds from Traffic API
    4. **Run A*** - Pathfinding algorithm with traffic-weighted edges
    5. **Calculate ETA** - ML model predicts arrival time
    6. **Cache result** - Store route for subsequent requests

    ---

    ### Traffic Processing Flow

    ```mermaid
    sequenceDiagram
        participant Mobile
        participant Location_Processor
        participant Kafka
        participant Traffic_ML
        participant Redis_Traffic
        participant Traffic_DB

        Mobile->>Kafka: GPS probe (lat, lng, speed, timestamp)
        Note over Mobile: 100M updates/sec

        Kafka->>Location_Processor: Consume location batch
        Location_Processor->>Location_Processor: Map match to road segments
        Location_Processor->>Location_Processor: Calculate segment speeds

        Location_Processor->>Redis_Traffic: UPDATE segment:12345 (current speed)
        Location_Processor->>Kafka: Publish traffic_update event

        Kafka->>Traffic_ML: Consume traffic updates
        Traffic_ML->>Traffic_ML: Run ML model (predict next 60 min)
        Traffic_ML->>Redis_Traffic: SET predicted_traffic:segment:12345

        Kafka->>Traffic_DB: Store historical data
        Note over Traffic_DB: Used for ML training

        Traffic_ML->>Traffic_DB: Query historical patterns
        Traffic_DB-->>Traffic_ML: Historical speeds (same day/time)
    ```

    **Flow Explanation:**

    1. **GPS probes** - Mobile apps send location updates every 10 seconds
    2. **Map matching** - Match GPS points to road segments
    3. **Speed calculation** - Calculate current speed for each segment
    4. **Update Redis** - Store current traffic state (5 GB in-memory)
    5. **ML prediction** - Predict traffic for next 60 minutes
    6. **Historical storage** - Archive for ML training and analysis

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Google Maps subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Routing Algorithm** | How to find fastest route in milliseconds? | A* with contraction hierarchies |
    | **Traffic Prediction** | How to predict traffic 60 minutes ahead? | ML model (XGBoost) with historical patterns |
    | **Map Tile Serving** | How to serve 579K tiles/sec globally? | CDN + pre-rendered tiles + quad-tree indexing |
    | **ETA Accuracy** | How to achieve 90% accuracy within ±5 min? | Ensemble models combining traffic, historical, events |

    ---

    === "🛣️ Routing Algorithm"

        ## The Challenge

        **Problem:** Find shortest path in graph with 100M segments in < 100ms. Classic Dijkstra is O(E log V), too slow!

        **Naive approach:** Run Dijkstra from origin to destination. **Doesn't scale** (500ms+ for cross-country routes).

        **Evolution:**

        1. **2005:** Pure Dijkstra - 500ms for long routes
        2. **2010:** Bidirectional Dijkstra - 250ms (search from both ends)
        3. **2015:** A* with landmarks - 100ms (heuristic guidance)
        4. **2020+:** Contraction Hierarchies - 10ms (pre-processed shortcuts)

        ---

        ## A* Algorithm with Contraction Hierarchies

        **Core Idea:** Pre-process graph to create "highway" shortcuts, then search only relevant portions.

        ### Contraction Hierarchies (Pre-processing)

        ```python
        class ContractionHierarchies:
            """
            Pre-process road network to create hierarchical shortcuts

            Key insight: Most routes use highways for long distances.
            Pre-compute shortcuts to skip local roads.
            """

            def __init__(self, graph):
                self.graph = graph  # Original road network
                self.shortcuts = {}  # Pre-computed shortcuts
                self.node_order = {}  # Importance ranking

            def preprocess(self):
                """
                Offline pre-processing: Contract graph into hierarchies

                Time: ~1 hour for entire road network
                Frequency: Once per week (when roads change)
                """
                # Step 1: Rank nodes by importance
                # Highways = high importance, local roads = low importance
                nodes = self.graph.get_all_nodes()

                for node in sorted(nodes, key=self._importance_score):
                    self._contract_node(node)

                logger.info(f"Created {len(self.shortcuts)} shortcuts")

            def _importance_score(self, node):
                """
                Calculate node importance for contraction order

                Factors:
                - Highway nodes contracted last (higher importance)
                - High-degree nodes contracted later
                - Spatial location (city centers = important)
                """
                neighbors = self.graph.get_neighbors(node)
                road_types = [n['road_type'] for n in neighbors]

                score = 0

                # Road type importance
                if 'highway' in road_types:
                    score += 1000
                elif 'primary' in road_types:
                    score += 500
                elif 'secondary' in road_types:
                    score += 100

                # Degree (number of connections)
                score += len(neighbors) * 10

                # Spatial centrality (simplified)
                if self._is_city_center(node):
                    score += 2000

                return score

            def _contract_node(self, node):
                """
                Contract node: create shortcuts bypassing this node

                Example:
                    A -> node -> B (distance: 5 + 7 = 12)
                    Create shortcut: A -> B (distance: 12)
                """
                incoming = self.graph.get_incoming_edges(node)
                outgoing = self.graph.get_outgoing_edges(node)

                # Try all combinations of incoming -> outgoing
                for in_edge in incoming:
                    for out_edge in outgoing:
                        # Calculate shortcut distance
                        shortcut_dist = in_edge['distance'] + out_edge['distance']

                        # Only create shortcut if it's useful
                        # (not redundant with existing edges)
                        if self._is_useful_shortcut(in_edge, out_edge, shortcut_dist):
                            self.shortcuts[(in_edge['from'], out_edge['to'])] = {
                                'distance': shortcut_dist,
                                'via': node,
                                'level': max(in_edge.get('level', 0), out_edge.get('level', 0)) + 1
                            }

            def _is_useful_shortcut(self, in_edge, out_edge, shortcut_dist):
                """
                Only create shortcut if it's the only path (witness search)
                """
                # Check if there's an alternative path that's shorter
                alt_paths = self.graph.find_alternative_paths(
                    in_edge['from'],
                    out_edge['to'],
                    max_distance=shortcut_dist,
                    exclude=[in_edge['via']]
                )

                # If no alternative path exists, shortcut is useful
                return len(alt_paths) == 0
        ```

        ---

        ## Bidirectional A* Search

        ```python
        import heapq
        from typing import List, Tuple, Dict
        import math

        class BidirectionalAStar:
            """
            Bidirectional A* with Contraction Hierarchies

            Search from both origin and destination simultaneously.
            Use pre-computed shortcuts to skip unimportant nodes.
            """

            def __init__(self, graph, shortcuts):
                self.graph = graph
                self.shortcuts = shortcuts

            def find_route(self, origin, destination, traffic_data=None):
                """
                Find fastest route using bidirectional A* with CH

                Args:
                    origin: Start node ID
                    destination: End node ID
                    traffic_data: Optional dict of segment -> current speed

                Returns:
                    Route object with path, distance, duration
                """
                # Priority queues for forward and backward search
                forward_queue = [(0, origin, [origin])]  # (f_score, node, path)
                backward_queue = [(0, destination, [destination])]

                # g_scores (cost from start/end)
                forward_g = {origin: 0}
                backward_g = {destination: 0}

                # Best known distance
                best_distance = float('inf')
                best_meeting_point = None

                # Search simultaneously
                while forward_queue and backward_queue:
                    # Expand forward
                    f_score, f_node, f_path = heapq.heappop(forward_queue)

                    # Check if we should stop (provably optimal)
                    if f_score >= best_distance:
                        break

                    # Explore neighbors (use shortcuts if available)
                    for neighbor, edge_data in self._get_neighbors_with_shortcuts(f_node, 'forward'):
                        # Calculate cost with traffic
                        edge_cost = self._calculate_edge_cost(edge_data, traffic_data)

                        new_g = forward_g[f_node] + edge_cost

                        if neighbor not in forward_g or new_g < forward_g[neighbor]:
                            forward_g[neighbor] = new_g

                            # A* heuristic: straight-line distance to destination
                            h_score = self._heuristic(neighbor, destination)
                            f_score_new = new_g + h_score

                            heapq.heappush(
                                forward_queue,
                                (f_score_new, neighbor, f_path + [neighbor])
                            )

                            # Check if forward and backward searches met
                            if neighbor in backward_g:
                                total_dist = new_g + backward_g[neighbor]
                                if total_dist < best_distance:
                                    best_distance = total_dist
                                    best_meeting_point = neighbor

                    # Expand backward (symmetric)
                    b_score, b_node, b_path = heapq.heappop(backward_queue)

                    if b_score >= best_distance:
                        break

                    for neighbor, edge_data in self._get_neighbors_with_shortcuts(b_node, 'backward'):
                        edge_cost = self._calculate_edge_cost(edge_data, traffic_data)
                        new_g = backward_g[b_node] + edge_cost

                        if neighbor not in backward_g or new_g < backward_g[neighbor]:
                            backward_g[neighbor] = new_g
                            h_score = self._heuristic(neighbor, origin)
                            b_score_new = new_g + h_score

                            heapq.heappush(
                                backward_queue,
                                (b_score_new, neighbor, [neighbor] + b_path)
                            )

                            if neighbor in forward_g:
                                total_dist = forward_g[neighbor] + new_g
                                if total_dist < best_distance:
                                    best_distance = total_dist
                                    best_meeting_point = neighbor

                # Reconstruct path
                if best_meeting_point:
                    return self._reconstruct_path(
                        origin,
                        best_meeting_point,
                        destination,
                        forward_g,
                        backward_g
                    )
                else:
                    return None  # No path found

            def _get_neighbors_with_shortcuts(self, node, direction):
                """
                Get neighbors, preferring shortcuts for high-level nodes

                Contraction Hierarchies rule: Only go UP the hierarchy
                - Forward search: only follow edges to higher-level nodes
                - Backward search: only follow edges from higher-level nodes
                """
                node_level = self.graph.get_node_level(node)

                # Get regular neighbors
                neighbors = self.graph.get_neighbors(node, direction)

                # Add shortcuts
                if direction == 'forward':
                    shortcuts = self.shortcuts.get(node, {})
                    for dest, shortcut_data in shortcuts.items():
                        if shortcut_data['level'] >= node_level:
                            neighbors.append((dest, shortcut_data))
                else:  # backward
                    # Find shortcuts ending at this node
                    for (src, dst), shortcut_data in self.shortcuts.items():
                        if dst == node and shortcut_data['level'] >= node_level:
                            neighbors.append((src, shortcut_data))

                return neighbors

            def _heuristic(self, node1, node2):
                """
                A* heuristic: straight-line distance divided by max speed

                Admissible: never overestimates actual cost
                """
                lat1, lng1 = self.graph.get_coordinates(node1)
                lat2, lng2 = self.graph.get_coordinates(node2)

                # Haversine distance
                distance_km = self._haversine_distance(lat1, lng1, lat2, lng2)

                # Assume max highway speed: 120 km/h
                min_time_hours = distance_km / 120.0

                return min_time_hours * 3600  # Convert to seconds

            def _haversine_distance(self, lat1, lng1, lat2, lng2):
                """Calculate great-circle distance between two points"""
                R = 6371  # Earth radius in km

                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)

                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dlng / 2) ** 2)

                c = 2 * math.asin(math.sqrt(a))

                return R * c

            def _calculate_edge_cost(self, edge_data, traffic_data):
                """
                Calculate edge traversal cost (time in seconds)

                Incorporates real-time traffic if available
                """
                segment_id = edge_data.get('segment_id')
                distance_m = edge_data['length_meters']

                # Get current speed from traffic data
                if traffic_data and segment_id in traffic_data:
                    current_speed_kmh = traffic_data[segment_id]['speed_kmh']
                else:
                    # Use historical average or speed limit
                    current_speed_kmh = edge_data.get('avg_speed_kmh', edge_data['speed_limit_kmh'])

                # Calculate travel time
                travel_time_sec = (distance_m / 1000.0) / current_speed_kmh * 3600

                return travel_time_sec

            def _reconstruct_path(self, origin, meeting_point, destination, forward_g, backward_g):
                """
                Reconstruct full path from origin to destination via meeting point
                """
                # This is simplified - actual implementation needs to unpack shortcuts
                forward_path = self._unpack_path(origin, meeting_point, forward_g)
                backward_path = self._unpack_path(meeting_point, destination, backward_g)

                full_path = forward_path + backward_path[1:]  # Avoid duplicating meeting point

                return {
                    'path': full_path,
                    'distance_meters': self._calculate_path_distance(full_path),
                    'duration_seconds': forward_g[meeting_point] + backward_g[meeting_point]
                }
        ```

        ---

        ## Routing Trade-offs

        | Algorithm | Time Complexity | Pre-processing | Memory | Use Case |
        |-----------|----------------|----------------|--------|----------|
        | **Dijkstra** | O(E log V) | None | O(V) | Small graphs (< 1K nodes) |
        | **A*** | O(E log V) | None | O(V) | Medium graphs (< 100K nodes) |
        | **Bidirectional A*** | O(E log V) | None | O(V) | General routing |
        | **Contraction Hierarchies** | O(log² V) | O(V²) once | O(V) | **Production (Google Maps)** |

        **Why CH wins:**

        - **10-100x faster** than pure A* (10ms vs 100ms)
        - **Scales to continental routes** (same performance for 10 mile or 1000 mile routes)
        - **Pre-processing parallelizable** (one-time cost, weekly updates)

    === "📊 Traffic Prediction"

        ## The Challenge

        **Problem:** Predict traffic conditions 60 minutes into the future with 90% accuracy.

        **Input data:**

        - **GPS probes:** 100M vehicles sending location every 10 seconds
        - **Historical patterns:** 2 years of speed data (10 TB compressed)
        - **External events:** Concerts, sports games, construction (API data)
        - **Weather:** Rain/snow impacts speed (weather API)

        **ML Model:** XGBoost ensemble with feature engineering

        ---

        ## Traffic ML Pipeline

        ```python
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        class TrafficPredictor:
            """
            ML-based traffic speed prediction

            Predicts speed for each road segment for next 60 minutes
            """

            def __init__(self, historical_db, event_api):
                self.historical_db = historical_db
                self.event_api = event_api
                self.models = {}  # One model per road segment type

            def train_models(self):
                """
                Train XGBoost models on historical data

                Frequency: Daily (incremental training)
                Training time: 4 hours on GPU cluster
                """
                # Query historical traffic data
                historical_data = self.historical_db.query("""
                    SELECT
                        segment_id,
                        timestamp,
                        speed_kmh,
                        day_of_week,
                        hour,
                        is_holiday,
                        weather_condition,
                        road_type
                    FROM traffic_speed
                    WHERE timestamp > now() - INTERVAL '90 days'
                """)

                df = pd.DataFrame(historical_data)

                # Feature engineering
                df = self._engineer_features(df)

                # Train separate models for different road types
                # (highways behave differently than city streets)
                for road_type in ['highway', 'primary', 'secondary', 'residential']:
                    road_df = df[df['road_type'] == road_type]

                    X = road_df.drop(['speed_kmh', 'timestamp', 'segment_id'], axis=1)
                    y = road_df['speed_kmh']

                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=500,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method='gpu_hist'  # GPU acceleration
                    )

                    model.fit(X, y)
                    self.models[road_type] = model

                    logger.info(f"Trained model for {road_type}: R² = {model.score(X, y):.3f}")

            def _engineer_features(self, df):
                """
                Create features from raw data

                Feature categories:
                1. Temporal: hour, day, week, month, is_holiday
                2. Historical: typical speed for this time/location
                3. Recent: speed trend (last 30 minutes)
                4. Spatial: nearby segment speeds
                5. External: weather, events
                """
                # 1. Temporal features
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                df['week_of_year'] = pd.to_datetime(df['timestamp']).dt.isocalendar().week
                df['month'] = pd.to_datetime(df['timestamp']).dt.month
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

                # 2. Historical features (typical speed for this segment/time)
                df['historical_speed_mean'] = df.groupby(['segment_id', 'day_of_week', 'hour'])['speed_kmh'].transform('mean')
                df['historical_speed_std'] = df.groupby(['segment_id', 'day_of_week', 'hour'])['speed_kmh'].transform('std')

                # 3. Recent trend features
                df = df.sort_values(['segment_id', 'timestamp'])
                df['speed_diff_15min'] = df.groupby('segment_id')['speed_kmh'].diff(periods=3)  # 15 min = 3 x 5min bins
                df['speed_diff_30min'] = df.groupby('segment_id')['speed_kmh'].diff(periods=6)

                # 4. Spatial features (average speed of nearby segments)
                # This requires joining with nearby segments - simplified here
                df['nearby_avg_speed'] = df['speed_kmh']  # Placeholder

                # 5. External features
                # Weather: encoded as categorical
                weather_map = {'clear': 0, 'rain': 1, 'snow': 2, 'fog': 3}
                df['weather_encoded'] = df['weather_condition'].map(weather_map)

                return df

            def predict_traffic(self, segment_id, prediction_time):
                """
                Predict speed for a segment at a future time

                Args:
                    segment_id: Road segment ID
                    prediction_time: Datetime to predict (up to 60 min in future)

                Returns:
                    Predicted speed in km/h
                """
                # Get segment metadata
                segment = self.graph.get_segment(segment_id)
                road_type = segment['road_type']

                # Get recent historical speed
                recent_speed = self.get_recent_speed(segment_id)

                # Get typical historical speed for this time
                typical_speed = self.historical_db.query_typical_speed(
                    segment_id,
                    prediction_time.weekday(),
                    prediction_time.hour
                )

                # Check for events near this segment
                events = self.event_api.get_nearby_events(
                    segment['lat'],
                    segment['lng'],
                    prediction_time,
                    radius_km=5
                )

                # Build feature vector
                features = {
                    'hour': prediction_time.hour,
                    'day_of_week': prediction_time.weekday(),
                    'week_of_year': prediction_time.isocalendar()[1],
                    'month': prediction_time.month,
                    'is_weekend': 1 if prediction_time.weekday() >= 5 else 0,
                    'is_rush_hour': 1 if prediction_time.hour in [7, 8, 9, 17, 18, 19] else 0,
                    'historical_speed_mean': typical_speed['mean'],
                    'historical_speed_std': typical_speed['std'],
                    'recent_speed': recent_speed,
                    'speed_diff_15min': recent_speed - typical_speed['mean'],
                    'nearby_avg_speed': self._get_nearby_avg_speed(segment_id),
                    'weather_encoded': self._get_weather_code(segment, prediction_time),
                    'has_event': 1 if len(events) > 0 else 0,
                    'is_holiday': self._is_holiday(prediction_time)
                }

                # Predict
                model = self.models.get(road_type, self.models['primary'])
                X = pd.DataFrame([features])

                predicted_speed = model.predict(X)[0]

                # Clamp to reasonable range
                min_speed = segment['speed_limit_kmh'] * 0.1  # At least 10% of speed limit
                max_speed = segment['speed_limit_kmh'] * 1.1  # At most 110% of speed limit

                return np.clip(predicted_speed, min_speed, max_speed)

            def predict_route_duration(self, route_segments, departure_time):
                """
                Predict total duration for a route considering traffic

                Args:
                    route_segments: List of segment IDs
                    departure_time: When route starts

                Returns:
                    Predicted total duration in seconds
                """
                total_duration = 0
                current_time = departure_time

                for segment_id in route_segments:
                    segment = self.graph.get_segment(segment_id)

                    # Predict speed at current_time
                    predicted_speed = self.predict_traffic(segment_id, current_time)

                    # Calculate segment duration
                    segment_duration = (segment['length_meters'] / 1000.0) / predicted_speed * 3600

                    total_duration += segment_duration

                    # Advance time for next segment
                    current_time += timedelta(seconds=segment_duration)

                return total_duration
        ```

        ---

        ## Real-time Processing (Kafka Streams)

        ```python
        from kafka import KafkaConsumer, KafkaProducer
        import json

        class RealTimeTrafficProcessor:
            """
            Process GPS probes in real-time to update traffic state

            Throughput: 100M location updates/sec
            Latency: < 1 second from GPS probe to traffic update
            """

            def __init__(self, kafka_brokers):
                self.consumer = KafkaConsumer(
                    'location_updates',
                    bootstrap_servers=kafka_brokers,
                    auto_offset_reset='latest',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )

                self.producer = KafkaProducer(
                    bootstrap_servers=kafka_brokers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )

                # In-memory aggregation state (per segment)
                self.segment_speeds = {}  # segment_id -> [speeds in last 60s]

            def process_stream(self):
                """
                Main processing loop

                Steps:
                1. Map match GPS to road segment
                2. Calculate speed
                3. Aggregate by segment (rolling window)
                4. Publish traffic updates
                """
                for message in self.consumer:
                    location_update = message.value

                    # Map match to road segment
                    segment_id = self._map_match(
                        location_update['lat'],
                        location_update['lng'],
                        location_update['heading']
                    )

                    if segment_id:
                        # Calculate speed (from GPS or accelerometer)
                        speed_kmh = location_update.get('speed_kmh', 0)

                        # Update aggregation window
                        if segment_id not in self.segment_speeds:
                            self.segment_speeds[segment_id] = []

                        self.segment_speeds[segment_id].append({
                            'speed': speed_kmh,
                            'timestamp': location_update['timestamp']
                        })

                        # Remove old data (> 60 seconds)
                        cutoff = location_update['timestamp'] - 60
                        self.segment_speeds[segment_id] = [
                            s for s in self.segment_speeds[segment_id]
                            if s['timestamp'] > cutoff
                        ]

                        # Calculate average speed
                        if len(self.segment_speeds[segment_id]) >= 5:  # Min 5 samples
                            avg_speed = np.mean([s['speed'] for s in self.segment_speeds[segment_id]])

                            # Publish traffic update
                            self.producer.send('traffic_updates', value={
                                'segment_id': segment_id,
                                'speed_kmh': avg_speed,
                                'sample_count': len(self.segment_speeds[segment_id]),
                                'timestamp': location_update['timestamp']
                            })

            def _map_match(self, lat, lng, heading):
                """
                Map match GPS point to road segment

                Algorithm: Find nearest segment within 50m, considering heading
                """
                # Query nearby segments (using spatial index)
                nearby_segments = self.spatial_index.query_radius(lat, lng, radius_m=50)

                if not nearby_segments:
                    return None

                # Find best match considering distance and heading
                best_segment = None
                best_score = 0

                for segment in nearby_segments:
                    # Distance score
                    dist = self._point_to_segment_distance(lat, lng, segment)
                    dist_score = 1.0 / (1.0 + dist)  # Closer = higher score

                    # Heading score (does GPS heading match segment direction?)
                    heading_diff = abs(heading - segment['bearing'])
                    heading_score = 1.0 - (heading_diff / 180.0)

                    # Combined score
                    score = dist_score * 0.7 + heading_score * 0.3

                    if score > best_score:
                        best_score = score
                        best_segment = segment

                return best_segment['segment_id'] if best_segment else None
        ```

        ---

        ## Traffic Prediction Trade-offs

        | Approach | Accuracy | Latency | Cost | Use Case |
        |----------|----------|---------|------|----------|
        | **Historical average** | 70% | 1ms | Low | Baseline |
        | **Linear regression** | 75% | 5ms | Low | Simple patterns |
        | **XGBoost** | 88% | 20ms | Medium | **Production (balanced)** |
        | **Deep learning (LSTM)** | 90% | 100ms | High | Research, critical routes |
        | **Ensemble (XGBoost + LSTM)** | 92% | 50ms | High | Premium features |

    === "🗺️ Map Tile Serving"

        ## The Challenge

        **Problem:** Serve 579K tile requests/sec globally with < 50ms latency.

        **Tile structure:**

        - **Zoom levels:** 0 (whole world) to 18 (street-level), 19 levels total
        - **Tile count:** 4^zoom tiles at each zoom (zoom 18 = 68 billion tiles)
        - **Tile size:** 256×256 pixels, ~10 KB PNG (compressed)
        - **Total storage:** 850 TB

        **Solutions:**

        1. **CDN caching:** 90% cache hit rate
        2. **Pre-rendering:** Offline tile generation
        3. **Quad-tree indexing:** Fast tile lookup
        4. **Dynamic tiles:** Generate on-demand for traffic/transit layers

        ---

        ## Tile Generation Pipeline

        ```python
        from PIL import Image, ImageDraw
        import math

        class MapTileGenerator:
            """
            Generate map tiles from vector data

            Pre-rendering: Weekly batch job for all tiles
            Dynamic rendering: On-demand for real-time layers (traffic)
            """

            def __init__(self, vector_db, s3_client):
                self.vector_db = vector_db  # PostGIS database with road vectors
                self.s3 = s3_client
                self.tile_size = 256

            def generate_tile(self, zoom, x, y):
                """
                Generate a single map tile

                Args:
                    zoom: Zoom level (0-18)
                    x: Tile X coordinate
                    y: Tile Y coordinate

                Returns:
                    PIL Image object
                """
                # Calculate geographic bounds for this tile
                bounds = self._tile_to_bounds(zoom, x, y)

                # Query vector data within bounds
                features = self.vector_db.query_features(
                    bounds,
                    zoom_level=zoom  # Simplification level depends on zoom
                )

                # Create image
                img = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
                draw = ImageDraw.Draw(img)

                # Render features by type (order matters for layering)
                feature_order = ['water', 'parks', 'buildings', 'roads', 'labels']

                for feature_type in feature_order:
                    type_features = [f for f in features if f['type'] == feature_type]
                    self._render_features(draw, type_features, bounds, zoom)

                return img

            def _tile_to_bounds(self, zoom, x, y):
                """
                Convert tile coordinates to geographic bounds (lat/lng)

                Tile coordinates use Web Mercator projection
                """
                n = 2 ** zoom

                # Northwest corner
                lng_min = x / n * 360.0 - 180.0
                lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
                lat_max = math.degrees(lat_rad_max)

                # Southeast corner
                lng_max = (x + 1) / n * 360.0 - 180.0
                lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
                lat_min = math.degrees(lat_rad_min)

                return {
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lng_min': lng_min,
                    'lng_max': lng_max
                }

            def _render_features(self, draw, features, bounds, zoom):
                """
                Render features on tile

                Different rendering based on zoom level:
                - Low zoom (0-8): Only major roads, cities
                - Medium zoom (9-13): More roads, labels
                - High zoom (14-18): All details, building outlines
                """
                for feature in features:
                    if feature['type'] == 'road':
                        self._render_road(draw, feature, bounds, zoom)
                    elif feature['type'] == 'water':
                        self._render_polygon(draw, feature, bounds, color='lightblue')
                    elif feature['type'] == 'building':
                        self._render_polygon(draw, feature, bounds, color='lightgray')
                    elif feature['type'] == 'label':
                        self._render_label(draw, feature, bounds, zoom)

            def _render_road(self, draw, road, bounds, zoom):
                """
                Render road as line with width based on road type and zoom
                """
                # Road width varies by type and zoom
                width_map = {
                    'highway': max(6, zoom - 8),
                    'primary': max(4, zoom - 10),
                    'secondary': max(2, zoom - 12),
                    'residential': max(1, zoom - 14)
                }

                road_type = road.get('road_type', 'residential')
                width = width_map.get(road_type, 1)

                # Only render if appropriate for zoom level
                if zoom < 10 and road_type not in ['highway']:
                    return  # Skip minor roads at low zoom

                # Convert coordinates to pixel positions
                coords = road['geometry']['coordinates']
                pixel_coords = [
                    self._latng_to_pixel(lat, lng, bounds)
                    for lng, lat in coords
                ]

                # Draw line
                if len(pixel_coords) >= 2:
                    draw.line(pixel_coords, fill='gray', width=width)

            def _latng_to_pixel(self, lat, lng, bounds):
                """Convert lat/lng to pixel position within tile"""
                x_percent = (lng - bounds['lng_min']) / (bounds['lng_max'] - bounds['lng_min'])
                y_percent = (bounds['lat_max'] - lat) / (bounds['lat_max'] - bounds['lat_min'])

                return (
                    int(x_percent * self.tile_size),
                    int(y_percent * self.tile_size)
                )

            def batch_generate_tiles(self, zoom_level):
                """
                Batch generate all tiles for a zoom level

                Parallelized across cluster

                Example: Zoom 18 = 68B tiles, ~680 TB
                Time: 100 hours on 1000-node cluster
                """
                n = 2 ** zoom_level

                logger.info(f"Generating {n*n} tiles for zoom {zoom_level}")

                for x in range(n):
                    for y in range(n):
                        # Generate tile
                        tile_img = self.generate_tile(zoom_level, x, y)

                        # Compress and save to S3
                        tile_bytes = self._compress_tile(tile_img)

                        s3_key = f"tiles/{zoom_level}/{x}/{y}.png"
                        self.s3.put_object(
                            Bucket='maps-tiles',
                            Key=s3_key,
                            Body=tile_bytes,
                            ContentType='image/png',
                            CacheControl='public, max-age=86400'  # 24 hour cache
                        )

                    if x % 100 == 0:
                        logger.info(f"Progress: {x}/{n} ({x/n*100:.1f}%)")

            def _compress_tile(self, img):
                """Compress tile to PNG bytes"""
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                return buffer.getvalue()
        ```

        ---

        ## Quad-tree Tile Indexing

        ```python
        class TileQuadTree:
            """
            Quad-tree index for efficient tile lookup

            Allows quick ancestor/descendant queries:
            - "Get parent tile" (zoom out)
            - "Get child tiles" (zoom in)
            - "Get siblings" (pan)
            """

            def __init__(self):
                self.tree = {}

            def insert(self, zoom, x, y, tile_url):
                """
                Insert tile into quad-tree

                Uses quadkey for indexing (Bing Maps convention)
                """
                quadkey = self._to_quadkey(zoom, x, y)
                self.tree[quadkey] = {
                    'zoom': zoom,
                    'x': x,
                    'y': y,
                    'url': tile_url
                }

            def _to_quadkey(self, zoom, x, y):
                """
                Convert tile coordinates to quadkey

                Quadkey: Base-4 string encoding tile position
                Example: zoom=2, x=1, y=1 -> "11"

                Benefits:
                - String prefix = ancestor tiles
                - Fast hierarchy traversal
                - Efficient storage in key-value stores
                """
                quadkey = []
                for i in range(zoom, 0, -1):
                    digit = 0
                    mask = 1 << (i - 1)

                    if (x & mask) != 0:
                        digit += 1
                    if (y & mask) != 0:
                        digit += 2

                    quadkey.append(str(digit))

                return ''.join(quadkey)

            def get_parent(self, zoom, x, y):
                """Get parent tile (one zoom level out)"""
                if zoom == 0:
                    return None

                parent_quadkey = self._to_quadkey(zoom, x, y)[:-1]
                return self.tree.get(parent_quadkey)

            def get_children(self, zoom, x, y):
                """Get 4 child tiles (one zoom level in)"""
                if zoom >= 18:
                    return []

                children = []
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        child_x = x * 2 + dx
                        child_y = y * 2 + dy
                        child_quadkey = self._to_quadkey(zoom + 1, child_x, child_y)

                        if child_quadkey in self.tree:
                            children.append(self.tree[child_quadkey])

                return children
        ```

        ---

        ## Dynamic Traffic Overlay Tiles

        ```python
        class TrafficOverlayGenerator:
            """
            Generate real-time traffic overlay tiles

            Updated every 30 seconds with current traffic
            """

            def __init__(self, traffic_service):
                self.traffic = traffic_service

            def generate_traffic_tile(self, zoom, x, y):
                """
                Generate transparent overlay showing traffic

                Color coding:
                - Green: Free flow (> 80% speed limit)
                - Yellow: Moderate (50-80%)
                - Orange: Heavy (25-50%)
                - Red: Stopped (< 25%)
                """
                bounds = self._tile_to_bounds(zoom, x, y)

                # Get traffic data for segments in bounds
                traffic_data = self.traffic.get_traffic_in_bounds(bounds)

                # Create transparent overlay
                img = Image.new('RGBA', (256, 256), color=(0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                for segment in traffic_data:
                    # Determine color based on congestion
                    color = self._congestion_color(
                        segment['current_speed'],
                        segment['free_flow_speed']
                    )

                    # Draw segment
                    coords = segment['geometry']
                    pixel_coords = [self._latng_to_pixel(lat, lng, bounds) for lng, lat in coords]

                    if len(pixel_coords) >= 2:
                        draw.line(pixel_coords, fill=color, width=4)

                return img

            def _congestion_color(self, current_speed, free_flow_speed):
                """
                Get color based on congestion level

                Returns RGBA tuple
                """
                ratio = current_speed / max(free_flow_speed, 1)

                if ratio > 0.8:
                    return (0, 255, 0, 200)  # Green
                elif ratio > 0.5:
                    return (255, 255, 0, 200)  # Yellow
                elif ratio > 0.25:
                    return (255, 165, 0, 200)  # Orange
                else:
                    return (255, 0, 0, 200)  # Red
        ```

    === "⏱️ ETA Accuracy"

        ## The Challenge

        **Problem:** Achieve 90% accuracy within ±5 minutes for ETA predictions.

        **Challenges:**

        - **Traffic variability:** Congestion unpredictable
        - **Long routes:** Errors compound over distance
        - **User behavior:** Stops, detours, speed variance
        - **External events:** Accidents, construction, weather

        **Solution:** Ensemble model combining multiple prediction approaches

        ---

        ## ETA Ensemble Model

        ```python
        class ETAPredictor:
            """
            Ensemble ETA prediction combining multiple models

            Components:
            1. Baseline: Free-flow time (no traffic)
            2. Historical: Typical time for this route/time
            3. Current traffic: Real-time traffic adjusted
            4. ML prediction: 60-min ahead traffic forecast
            5. User behavior: Driver-specific adjustments
            """

            def __init__(self, traffic_predictor, historical_db, user_db):
                self.traffic = traffic_predictor
                self.historical = historical_db
                self.users = user_db

                # Ensemble weights (learned from validation data)
                self.weights = {
                    'baseline': 0.1,
                    'historical': 0.2,
                    'current_traffic': 0.3,
                    'ml_prediction': 0.3,
                    'user_behavior': 0.1
                }

            def predict_eta(self, route, user_id, departure_time):
                """
                Predict ETA for a route using ensemble

                Args:
                    route: List of road segments
                    user_id: User making the trip (for personalization)
                    departure_time: When trip starts

                Returns:
                    {
                        'eta_seconds': predicted ETA,
                        'confidence_interval': (min, max),
                        'breakdown': {component: duration}
                    }
                """
                predictions = {}

                # 1. Baseline (free-flow)
                predictions['baseline'] = self._baseline_eta(route)

                # 2. Historical (typical for this route/time)
                predictions['historical'] = self._historical_eta(
                    route,
                    departure_time
                )

                # 3. Current traffic
                predictions['current_traffic'] = self._current_traffic_eta(
                    route,
                    departure_time
                )

                # 4. ML prediction (future traffic)
                predictions['ml_prediction'] = self._ml_prediction_eta(
                    route,
                    departure_time
                )

                # 5. User behavior adjustment
                predictions['user_behavior'] = self._user_behavior_adjustment(
                    route,
                    user_id,
                    predictions['ml_prediction']
                )

                # Ensemble: Weighted average
                eta_seconds = sum(
                    predictions[component] * self.weights[component]
                    for component in predictions
                )

                # Calculate confidence interval
                variance = np.var(list(predictions.values()))
                std_dev = np.sqrt(variance)

                confidence_interval = (
                    eta_seconds - 1.96 * std_dev,  # 95% CI lower bound
                    eta_seconds + 1.96 * std_dev   # 95% CI upper bound
                )

                return {
                    'eta_seconds': int(eta_seconds),
                    'arrival_time': departure_time + timedelta(seconds=eta_seconds),
                    'confidence_interval': confidence_interval,
                    'breakdown': predictions
                }

            def _baseline_eta(self, route):
                """
                Baseline: Free-flow time (speed limit, no traffic)
                """
                total_time = 0

                for segment in route:
                    segment_time = (
                        segment['length_meters'] / 1000.0 /
                        segment['speed_limit_kmh'] * 3600
                    )
                    total_time += segment_time

                return total_time

            def _historical_eta(self, route, departure_time):
                """
                Historical: Typical time for this route at this time

                Query historical trip data for similar routes/times
                """
                # Simplification: Use average historical speed for each segment
                total_time = 0

                for segment in route:
                    historical_speed = self.historical.get_typical_speed(
                        segment['segment_id'],
                        departure_time.weekday(),
                        departure_time.hour
                    )

                    segment_time = (
                        segment['length_meters'] / 1000.0 /
                        historical_speed * 3600
                    )
                    total_time += segment_time

                return total_time

            def _current_traffic_eta(self, route, departure_time):
                """
                Current traffic: Use current speeds, assume stable
                """
                total_time = 0

                for segment in route:
                    current_speed = self.traffic.get_current_speed(segment['segment_id'])

                    segment_time = (
                        segment['length_meters'] / 1000.0 /
                        current_speed * 3600
                    )
                    total_time += segment_time

                return total_time

            def _ml_prediction_eta(self, route, departure_time):
                """
                ML prediction: Predict traffic for each segment at arrival time

                Most accurate for near-term (< 60 min) predictions
                """
                total_time = 0
                current_time = departure_time

                for segment in route:
                    # Predict speed at current_time (when we'll reach this segment)
                    predicted_speed = self.traffic.predict_traffic(
                        segment['segment_id'],
                        current_time
                    )

                    segment_time = (
                        segment['length_meters'] / 1000.0 /
                        predicted_speed * 3600
                    )

                    total_time += segment_time
                    current_time += timedelta(seconds=segment_time)

                return total_time

            def _user_behavior_adjustment(self, route, user_id, base_eta):
                """
                User behavior: Personalize based on driver habits

                Factors:
                - Typical speed (aggressive vs cautious driver)
                - Stop frequency (errands, gas, etc.)
                - Route familiarity (faster on known routes)
                """
                user_profile = self.users.get_driving_profile(user_id)

                # Speed adjustment (e.g., user typically drives 5% faster than average)
                speed_factor = user_profile.get('speed_factor', 1.0)
                adjusted_eta = base_eta / speed_factor

                # Stop probability (e.g., 10% chance of 5-min stop for long trips)
                route_distance = sum(s['length_meters'] for s in route) / 1000.0

                if route_distance > 50:  # Long trip
                    stop_probability = 0.1
                    avg_stop_duration = 300  # 5 minutes
                    expected_stop_time = stop_probability * avg_stop_duration
                    adjusted_eta += expected_stop_time

                return adjusted_eta

            def update_with_realtime_position(self, route, user_id, current_position, elapsed_time):
                """
                Update ETA mid-trip based on actual progress

                As user drives, we can recalibrate prediction
                """
                # Find current segment
                current_segment_idx = self._find_current_segment(route, current_position)

                # Remaining route
                remaining_route = route[current_segment_idx:]

                # Recalculate ETA for remaining route
                remaining_eta = self.predict_eta(
                    remaining_route,
                    user_id,
                    datetime.utcnow()
                )

                # Compare actual vs predicted progress
                # (could be ahead or behind schedule)
                predicted_elapsed = self._get_predicted_elapsed(route[:current_segment_idx])
                actual_vs_predicted_ratio = elapsed_time / max(predicted_elapsed, 1)

                # Adjust remaining ETA based on actual progress
                adjusted_remaining_eta = remaining_eta['eta_seconds'] * actual_vs_predicted_ratio

                return {
                    'eta_seconds': int(adjusted_remaining_eta),
                    'arrival_time': datetime.utcnow() + timedelta(seconds=adjusted_remaining_eta),
                    'on_time_status': 'ahead' if actual_vs_predicted_ratio < 1.0 else 'behind'
                }
        ```

        ---

        ## ETA Accuracy Monitoring

        ```python
        class ETAAccuracyMonitor:
            """
            Monitor ETA prediction accuracy for continuous improvement

            Metrics:
            - Mean Absolute Error (MAE)
            - Root Mean Square Error (RMSE)
            - Percentage within ±5 minutes
            - Percentage within ±10%
            """

            def __init__(self, db):
                self.db = db

            def record_trip(self, trip_id, predicted_eta, actual_duration):
                """
                Record trip for accuracy analysis
                """
                self.db.insert_trip_record({
                    'trip_id': trip_id,
                    'predicted_eta': predicted_eta,
                    'actual_duration': actual_duration,
                    'error': abs(predicted_eta - actual_duration),
                    'error_percentage': abs(predicted_eta - actual_duration) / actual_duration * 100,
                    'timestamp': datetime.utcnow()
                })

            def calculate_accuracy_metrics(self, time_window_hours=24):
                """
                Calculate accuracy metrics over recent trips
                """
                trips = self.db.get_recent_trips(time_window_hours)

                errors = [t['error'] for t in trips]
                error_percentages = [t['error_percentage'] for t in trips]

                # Mean Absolute Error
                mae = np.mean(errors)

                # Root Mean Square Error
                rmse = np.sqrt(np.mean([e**2 for e in errors]))

                # Percentage within ±5 minutes
                within_5min = len([e for e in errors if e <= 300]) / len(errors) * 100

                # Percentage within ±10%
                within_10pct = len([e for e in error_percentages if e <= 10]) / len(errors) * 100

                return {
                    'mae_seconds': mae,
                    'rmse_seconds': rmse,
                    'within_5min_percent': within_5min,
                    'within_10pct_percent': within_10pct,
                    'sample_size': len(trips)
                }
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Google Maps from 1M to 1B users.

    **Scaling challenges at 1B users:**

    - **Write throughput:** 110M location updates/sec
    - **Read throughput:** 596K tile requests/sec
    - **Route computation:** 35K routes/sec with < 100ms latency
    - **Storage:** 866 TB of map data
    - **Global distribution:** Sub-100ms latency worldwide

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Routing engine** | ✅ Yes | Contraction hierarchies (10ms routes), cache common routes |
    | **Tile serving** | ✅ Yes | CDN with 90% hit rate, 20 global edge locations |
    | **Location ingestion** | ✅ Yes | Kafka with 100 partitions, batch processing (1000 events/batch) |
    | **Traffic ML** | ✅ Yes | GPU clusters, model inference caching, 5-min update cycle |
    | **Graph database** | 🟡 Approaching | Read replicas (10 regions), in-memory cache (60 GB) |

    ---

    ## Geographic Partitioning

    **Problem:** Single datacenter can't serve 1B users globally with < 100ms latency.

    **Solution:** Multi-region deployment with geographic routing

    ```
    Global Infrastructure:
    ├── 10 Regional Datacenters
    │   ├── North America (2): US-East, US-West
    │   ├── Europe (2): EU-West, EU-Central
    │   ├── Asia (4): Asia-East, Asia-South, Asia-Southeast, Asia-Northeast
    │   ├── South America (1): SA-East
    │   └── Oceania (1): Australia
    │
    ├── 20 Edge Locations (CDN PoPs)
    │   └── Serve tiles with < 20ms latency
    │
    └── Data Replication
        ├── Road graph: Replicated to all regions (60 GB each)
        ├── Traffic data: Regional (only local traffic)
        └── Tiles: CDN-cached globally
    ```

    **Traffic Routing:**

    ```python
    def route_request(user_lat, user_lng):
        """
        Route user to nearest datacenter

        Uses GeoDNS + load balancer
        """
        # Determine nearest region
        region = get_nearest_region(user_lat, user_lng)

        # Route to regional endpoint
        return f"maps-api.{region}.google.com"
    ```

    ---

    ## Caching Strategy

    **Multi-tier caching:**

    | Layer | Technology | Size | TTL | Hit Rate |
    |-------|-----------|------|-----|----------|
    | **Browser cache** | LocalStorage | 50 MB | 7 days | 60% (repeat views) |
    | **CDN cache** | Cloudflare/Akamai | 10 TB/PoP | 24 hours | 90% (popular tiles) |
    | **API cache (Redis)** | Redis Cluster | 1.4 TB | 5-60 min | 70% (routes, traffic) |
    | **Database cache** | In-memory | 60 GB | N/A | 95% (graph queries) |

    **Cache invalidation:**

    ```python
    class CacheInvalidation:
        """
        Invalidate caches when data changes
        """

        def on_road_update(self, segment_id):
            """
            When road data changes (construction, new roads)
            """
            # Invalidate graph cache
            self.redis.delete(f"graph:segment:{segment_id}")

            # Invalidate affected tiles
            affected_tiles = self.get_tiles_containing_segment(segment_id)
            for tile in affected_tiles:
                self.cdn.purge(f"tiles/{tile.zoom}/{tile.x}/{tile.y}.png")

        def on_traffic_update(self, segment_id):
            """
            When traffic changes (every 30 seconds)
            """
            # Update Redis traffic cache
            self.redis.setex(
                f"traffic:segment:{segment_id}",
                60,  # 60 second TTL
                traffic_data
            )

            # Invalidate traffic overlay tiles
            affected_tiles = self.get_tiles_containing_segment(segment_id)
            for tile in affected_tiles:
                self.cdn.purge(f"traffic-overlay/{tile.zoom}/{tile.x}/{tile.y}.png")
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 1B users:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $432,000 (3,000 × c5.2xlarge) |
    | **Graph DB (Neo4j)** | $216,000 (1,800 × r5.large, 10 regions) |
    | **Kafka cluster** | $108,000 (1,000 brokers) |
    | **Redis cache** | $86,400 (400 nodes) |
    | **InfluxDB (traffic)** | $54,000 (500 nodes) |
    | **PostgreSQL (places/users)** | $21,600 (200 nodes) |
    | **S3 storage (tiles)** | $19,800 (850 TB) |
    | **CDN** | $595,000 (7,000 TB egress) |
    | **GPU compute (ML training)** | $108,000 (150 × p3.2xlarge) |
    | **Data transfer** | $180,000 (inter-region replication) |
    | **Total** | **$1,821,000/month** |

    **Cost optimizations:**

    - **Tile compression:** 10 KB → 5 KB (50% savings on CDN)
    - **Cold storage:** Move old tiles to Glacier (90% cheaper)
    - **Spot instances:** ML training on spot (70% cheaper)
    - **Traffic prediction:** Cache predictions (reduce GPU inference by 80%)

    ---

    ## Performance Optimizations

    ### Route Calculation

    ```python
    # Optimization 1: Route caching
    # Cache top 10M routes (e.g., home to work)
    # Cache hit rate: 20%

    route_cache_key = f"route:{origin}:{destination}:{departure_hour}"
    cached_route = redis.get(route_cache_key)

    if cached_route:
        # Just update ETA with current traffic
        return update_eta(cached_route)
    else:
        # Compute route
        route = compute_route(origin, destination)
        redis.setex(route_cache_key, 300, route)  # 5 min TTL
        return route

    # Optimization 2: Parallel alternative routes
    # Compute 3 alternative routes in parallel

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(compute_route, origin, destination, avoid='highways'),
            executor.submit(compute_route, origin, destination, optimize='shortest'),
            executor.submit(compute_route, origin, destination, optimize='fastest')
        ]

        routes = [f.result() for f in as_completed(futures)]
    ```

    ### Map Tile Serving

    ```python
    # Optimization 1: Vector tiles (smaller than raster)
    # Raster (PNG): 10 KB
    # Vector (PBF): 3 KB
    # Savings: 70%

    # Optimization 2: Tile request batching
    # Client requests 9 tiles at once (3×3 grid)
    # Single HTTP/2 request instead of 9

    # Optimization 3: Predictive prefetching
    # Pre-load tiles user is likely to pan to

    def prefetch_tiles(current_zoom, current_x, current_y, user_heading):
        """
        Prefetch tiles in direction of user movement
        """
        # Predict next tiles based on heading
        if user_heading == 'north':
            prefetch = [(current_x, current_y - 1)]
        elif user_heading == 'east':
            prefetch = [(current_x + 1, current_y)]
        # ... etc

        # Fetch in background
        for x, y in prefetch:
            asyncio.create_task(fetch_tile(current_zoom, x, y))
    ```

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Route Latency (P95)** | < 100ms | > 200ms |
    | **Tile Latency (P95)** | < 50ms | > 100ms |
    | **ETA Accuracy (24h)** | 90% within ±5 min | < 85% |
    | **CDN Cache Hit Rate** | > 90% | < 85% |
    | **Location Ingestion Lag** | < 1s | > 5s |
    | **Graph DB Query Time** | < 10ms | > 50ms |

    **Alerting rules:**

    ```yaml
    alerts:
      - name: HighRouteLatency
        condition: route_latency_p95 > 200ms
        severity: critical
        action: Scale routing cluster, check graph cache

      - name: LowETAAccuracy
        condition: eta_accuracy_5min < 85%
        severity: high
        action: Retrain ML models, check traffic data pipeline

      - name: CDNCacheHitRateDrop
        condition: cdn_cache_hit_rate < 85%
        severity: medium
        action: Check cache invalidation, increase cache size
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Contraction Hierarchies:** Pre-process graph for 10ms routing
    2. **A* algorithm:** Bidirectional search with traffic weighting
    3. **ML traffic prediction:** XGBoost ensemble for 88% accuracy
    4. **CDN tile serving:** 90% cache hit rate, 20 global PoPs
    5. **ETA ensemble:** Combine 5 models for 90% accuracy within ±5 min
    6. **Geographic partitioning:** 10 regional datacenters for < 100ms latency

    ---

    ## Interview Tips

    ✅ **Start with routing algorithm** - Core functionality, discuss A* vs Dijkstra

    ✅ **Emphasize scale** - 110M location updates/sec, 579K tile requests/sec

    ✅ **Discuss trade-offs** - Pre-computation (CH) vs dynamic (pure A*)

    ✅ **ML for traffic** - Historical patterns + real-time data

    ✅ **CDN critical** - 90% of traffic served from edge

    ✅ **ETA accuracy** - Business-critical, multi-model ensemble

    ✅ **Geospatial indexing** - S2, quad-tree for spatial queries

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How does A* work?"** | Priority queue, heuristic (straight-line distance), f(n) = g(n) + h(n) |
    | **"Why contraction hierarchies?"** | 10-100x faster, pre-compute shortcuts, hierarchical graph |
    | **"How to predict traffic?"** | ML model (XGBoost), features: time, historical, weather, events |
    | **"How to serve tiles fast?"** | CDN caching (90% hit rate), pre-rendered tiles, vector tiles |
    | **"How to handle 110M location updates/sec?"** | Kafka partitioning, batch processing, map matching |
    | **"How to make ETA accurate?"** | Ensemble model, update mid-trip, user behavior personalization |
    | **"How to store road network?"** | Graph database (Neo4j), in-memory cache, geographic partitioning |

    ---

    ## Advanced Topics

    **If time permits, discuss:**

    1. **Offline maps:** Pre-download tiles + graph for region, local routing
    2. **Multi-modal routing:** Combine driving + walking + transit
    3. **Real-time rerouting:** Detect user off route, recalculate instantly
    4. **Map matching:** Snap GPS traces to road network (HMM algorithm)
    5. **Tile compression:** Vector tiles (70% smaller than raster)
    6. **Graph updates:** Incremental updates for new roads, closures

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Google, Apple, TomTom, HERE, Waze, Uber, Lyft

---

*Master this problem and you'll be ready for: Uber, Lyft, DoorDash, Waze, HERE Technologies, autonomous vehicle routing*
