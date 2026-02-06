# Design a Smart City Traffic Management System

An intelligent traffic management platform that optimizes traffic flow using IoT sensors, ML-based predictions, adaptive signal timing, and real-time incident detection across city-wide infrastructure.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐ Medium | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K intersections, 1M vehicles, 100K sensors, 500K requests/sec, 50 cities |
| **Key Challenges** | Real-time signal optimization, congestion prediction, incident detection, traffic flow analysis, multi-city coordination |
| **Core Concepts** | IoT sensor networks, reinforcement learning, LSTM for prediction, computer vision, edge computing, graph theory |
| **Companies** | Siemens Mobility, IBM Smart Cities, Cisco Smart+Connected Communities, GE Transportation, City governments |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Traffic Signal Control** | Adaptive signal timing based on real-time traffic | P0 (Must have) |
    | **Congestion Prediction** | Predict traffic congestion 30-60 minutes ahead | P0 (Must have) |
    | **Incident Detection** | Detect accidents, breakdowns, road hazards | P0 (Must have) |
    | **Traffic Flow Analysis** | Monitor vehicle flow, speed, density | P0 (Must have) |
    | **Real-time Routing** | Suggest optimal routes based on current conditions | P0 (Must have) |
    | **Emergency Vehicle Priority** | Clear paths for ambulances, fire trucks | P0 (Must have) |
    | **Analytics Dashboard** | Real-time monitoring and historical analytics | P1 (Should have) |
    | **Public Transit Integration** | Coordinate with bus/train schedules | P1 (Should have) |
    | **Parking Management** | Guide drivers to available parking spots | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Vehicle-to-vehicle communication (V2V)
    - Autonomous vehicle coordination
    - Toll collection systems
    - Traffic violation enforcement
    - Weather prediction systems

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Traffic system failure can cause gridlock |
    | **Latency (Signal Control)** | < 100ms signal adjustments | Real-time response to traffic changes |
    | **Latency (Prediction)** | < 2s for prediction queries | Users need quick routing decisions |
    | **Accuracy (Prediction)** | > 85% accuracy | Poor predictions lead to increased congestion |
    | **Scalability** | 100K sensors per city | Support mega-cities (NYC, Tokyo, Mumbai) |
    | **Reliability** | Graceful degradation | Continue basic operations during failures |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total intersections: 10,000 per city
    Total vehicles tracked: 1,000,000 active vehicles
    Total sensors: 100,000 (traffic cameras, inductive loops, radars)
    Cities supported: 50 cities globally

    Sensor updates:
    - Each sensor reports every 5 seconds
    - 100K sensors × 0.2 updates/sec = 20K updates/sec
    - Across 50 cities: 1M updates/sec

    Signal adjustments:
    - 10K intersections × 4 signals = 40K signals
    - Adjusted every 30 seconds on average
    - 40K / 30 = 1.3K adjustments/sec

    Vehicle tracking:
    - GPS updates every 10 seconds
    - 1M vehicles × 0.1 updates/sec = 100K updates/sec
    - Across 50 cities: 5M updates/sec

    API requests (routing, predictions):
    - 10M users × 50 requests/day = 500M requests/day
    - Average: 5,800 requests/sec
    - Peak (rush hour): 17,400 requests/sec
    ```

    ### Storage Estimates

    ```
    Sensor data:
    - Per reading: 200 bytes (sensor_id, location, speed, volume, timestamp)
    - Daily: 1M updates/sec × 86,400 = 86.4B readings/day
    - Storage: 86.4B × 200 bytes = 17.3 TB/day
    - With retention (90 days): 1,557 TB = 1.56 PB

    Traffic incidents:
    - Per incident: 5 KB (location, type, severity, images, timestamps)
    - Daily incidents: 1,000 per city × 50 cities = 50K/day
    - Storage: 50K × 5 KB = 250 MB/day
    - Yearly: 91 GB

    Signal timing configurations:
    - Per intersection: 2 KB (timing plans, optimization parameters)
    - 10K intersections × 50 cities = 500K intersections
    - Storage: 500K × 2 KB = 1 GB

    Vehicle trajectories:
    - Per vehicle: 50 bytes (vehicle_id, lat, lng, speed, timestamp)
    - Daily: 5M updates/sec × 86,400 = 432B updates/day
    - Storage: 432B × 50 bytes = 21.6 TB/day
    - With retention (30 days): 648 TB

    Total: 1.56 PB (sensors) + 648 TB (vehicles) + 91 GB (incidents) + 1 GB (configs) ≈ 2.2 PB
    ```

    ### Memory Estimates (Caching)

    ```
    Current sensor states:
    - 5M sensors (50 cities × 100K) × 500 bytes = 2.5 GB

    Active vehicle locations:
    - 50M vehicles (50 cities × 1M) × 100 bytes = 5 GB

    Signal timing plans:
    - 500K intersections × 2 KB = 1 GB

    ML model cache:
    - Traffic prediction models: 50 cities × 100 MB = 5 GB
    - Incident detection models: 2 GB

    Real-time traffic graph:
    - Road network: 50 cities × 500 MB = 25 GB

    Total cache: 40.5 GB (distributed across cluster)
    ```

    ---

    ## Key Assumptions

    1. Average vehicle speed in city: 30 km/h (20 mph)
    2. Sensor accuracy: 95% for vehicle detection
    3. Average intersection delay: 45 seconds
    4. Peak to average traffic ratio: 3:1 (rush hour vs off-peak)
    5. Incident detection rate: > 90% within 2 minutes
    6. Green wave efficiency: 30% improvement in optimal conditions

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Edge computing** - Process sensor data at intersection level
    2. **ML-driven optimization** - Reinforcement learning for signal timing
    3. **Event-driven architecture** - Real-time incident response
    4. **Microservices** - Separate services for prediction, routing, control
    5. **Hierarchical control** - Intersection → Zone → City → Multi-city

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Edge Layer - Intersections"
            Sensor1[Traffic Sensors<br/>Cameras, Loops, Radar]
            EdgeComp1[Edge Computing Node<br/>Signal Controller]
            Signal1[Traffic Signals<br/>Adaptive Timing]
        end

        subgraph "Data Ingestion"
            StreamProc[Stream Processor<br/>Kafka/Flink<br/>1M events/sec]
            DataValid[Data Validation<br/>Anomaly detection]
        end

        subgraph "Core Services"
            TrafficAnalyzer[Traffic Flow Analyzer<br/>Speed, density, volume]
            MLPredictor[ML Prediction Service<br/>LSTM for congestion]
            SignalOptimizer[Signal Optimizer<br/>Reinforcement Learning]
            IncidentDetector[Incident Detector<br/>Computer Vision]
            RoutingService[Routing Service<br/>Dynamic path planning]
            EmergencyMgmt[Emergency Management<br/>Vehicle priority]
        end

        subgraph "ML Models"
            CongestionModel[Congestion Predictor<br/>LSTM/Transformer]
            IncidentModel[Incident Classifier<br/>CNN/YOLO]
            OptimizationModel[Signal RL Agent<br/>PPO/DQN]
        end

        subgraph "Storage"
            TimeSeries[(Time-Series DB<br/>InfluxDB/TimescaleDB<br/>Sensor data)]
            GraphDB[(Graph DB<br/>Neo4j<br/>Road network)]
            RelationalDB[(PostgreSQL<br/>Incidents, configs)]
            ObjectStore[(S3<br/>Camera footage)]
        end

        subgraph "Caching"
            RedisCache[Redis<br/>Real-time state]
            RedisGraph[Redis Graph<br/>Traffic network]
        end

        subgraph "Analytics & Visualization"
            Dashboard[Real-time Dashboard<br/>Traffic monitoring]
            Analytics[Analytics Engine<br/>Historical analysis]
            Reporting[Reporting Service<br/>Performance metrics]
        end

        subgraph "External Services"
            Maps[Maps API<br/>Base maps]
            Weather[Weather API<br/>Conditions data]
            Transit[Public Transit API<br/>Bus/train schedules]
        end

        Sensor1 --> EdgeComp1
        EdgeComp1 --> Signal1
        EdgeComp1 --> StreamProc

        StreamProc --> DataValid
        DataValid --> TrafficAnalyzer
        DataValid --> IncidentDetector
        DataValid --> TimeSeries

        TrafficAnalyzer --> MLPredictor
        TrafficAnalyzer --> RedisCache
        TrafficAnalyzer --> GraphDB

        MLPredictor --> CongestionModel
        MLPredictor --> RoutingService
        MLPredictor --> SignalOptimizer

        IncidentDetector --> IncidentModel
        IncidentDetector --> ObjectStore
        IncidentDetector --> RelationalDB
        IncidentDetector --> EmergencyMgmt

        SignalOptimizer --> OptimizationModel
        SignalOptimizer --> EdgeComp1

        RoutingService --> RedisGraph
        RoutingService --> Maps

        EmergencyMgmt --> SignalOptimizer

        TrafficAnalyzer --> Dashboard
        Analytics --> TimeSeries
        Analytics --> RelationalDB
        Analytics --> Reporting

        style StreamProc fill:#e8eaf6
        style RedisCache fill:#fff4e1
        style RedisGraph fill:#fff4e1
        style TimeSeries fill:#ffe1e1
        style GraphDB fill:#ffe1e1
        style RelationalDB fill:#ffe1e1
        style ObjectStore fill:#ffe1e1
        style CongestionModel fill:#e8f5e9
        style IncidentModel fill:#e8f5e9
        style OptimizationModel fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Edge Computing** | Process data locally, reduce latency (< 100ms) | Cloud-only (too slow, network dependency) |
    | **Kafka/Flink** | Handle 1M events/sec with exactly-once processing | RabbitMQ (lower throughput), direct DB writes (too slow) |
    | **InfluxDB** | Optimized for time-series sensor data (17TB/day) | PostgreSQL (poor performance), Cassandra (over-engineered) |
    | **Neo4j** | Efficient graph queries for road network routing | PostgreSQL with PostGIS (slower for graph traversal) |
    | **LSTM Models** | Capture temporal patterns in traffic (time-series) | Simple ML models (miss temporal dependencies) |
    | **Reinforcement Learning** | Optimize multi-intersection coordination | Rule-based (inflexible), genetic algorithms (slow convergence) |

    **Key Trade-off:** We chose **edge computing for signal control** (faster response) but **centralized ML training** (better models with more data). Hybrid approach balances latency and accuracy.

    ---

    ## API Design

    ### 1. Get Traffic Prediction

    **Request:**
    ```http
    POST /api/v1/traffic/predict
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "location": {
        "lat": 40.7580,
        "lng": -73.9855
      },
      "radius_km": 5,
      "prediction_horizon_minutes": 30,
      "timestamp": "2024-02-05T17:00:00Z"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "prediction_id": "pred_abc123",
      "timestamp": "2024-02-05T17:00:00Z",
      "predictions": [
        {
          "time": "2024-02-05T17:30:00Z",
          "congestion_level": "heavy",
          "average_speed_kmh": 15,
          "confidence": 0.87
        },
        {
          "time": "2024-02-05T18:00:00Z",
          "congestion_level": "moderate",
          "average_speed_kmh": 25,
          "confidence": 0.82
        }
      ],
      "affected_roads": [
        {
          "road_id": "road_123",
          "name": "Broadway",
          "predicted_delay_minutes": 12
        }
      ]
    }
    ```

    ---

    ### 2. Report Incident

    **Request:**
    ```http
    POST /api/v1/incidents
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "type": "accident",  // accident, breakdown, road_hazard, construction
      "location": {
        "lat": 40.7580,
        "lng": -73.9855,
        "road_id": "road_123"
      },
      "severity": "high",  // low, medium, high, critical
      "description": "Multi-vehicle accident blocking 2 lanes",
      "images": ["https://s3.../image1.jpg"],
      "reporter_id": "user_xyz789"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "incident_id": "inc_def456",
      "status": "confirmed",
      "estimated_clearance_time": "2024-02-05T18:30:00Z",
      "affected_routes": [
        {
          "route_id": "route_abc",
          "additional_delay_minutes": 15
        }
      ],
      "emergency_response_dispatched": true
    }
    ```

    ---

    ### 3. Optimize Signal Timing (Internal API)

    **Request:**
    ```http
    POST /internal/api/v1/signals/optimize
    Content-Type: application/json

    {
      "intersection_id": "int_123",
      "current_state": {
        "north_south_green": 45,
        "east_west_green": 35,
        "cycle_length": 90
      },
      "traffic_conditions": {
        "north_volume": 120,
        "south_volume": 95,
        "east_volume": 80,
        "west_volume": 75
      },
      "constraints": {
        "min_green_time": 15,
        "max_green_time": 90,
        "pedestrian_crossing_time": 8
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "optimized_timing": {
        "north_south_green": 55,
        "east_west_green": 30,
        "cycle_length": 95
      },
      "expected_improvement": {
        "delay_reduction_percent": 23,
        "throughput_increase_percent": 15
      },
      "confidence": 0.91,
      "apply_immediately": true
    }
    ```

    ---

    ## Database Schema

    ### Sensor Readings (InfluxDB - Time Series)

    ```sql
    -- Time-series measurement for sensor data
    CREATE MEASUREMENT sensor_readings (
        time TIMESTAMP,
        sensor_id TAG,
        intersection_id TAG,
        sensor_type TAG,  -- camera, loop, radar, lidar
        city_id TAG,

        -- Fields
        vehicle_count INTEGER,
        average_speed FLOAT,
        occupancy_percent FLOAT,
        queue_length INTEGER,

        -- Indexed by time + tags for efficient queries
        PRIMARY KEY (time, sensor_id)
    );

    -- Example query: Last 1 hour of data for an intersection
    SELECT
        time,
        sensor_id,
        vehicle_count,
        average_speed
    FROM sensor_readings
    WHERE
        intersection_id = 'int_123'
        AND time > NOW() - 1h
    ORDER BY time DESC;
    ```

    ---

    ### Road Network (Neo4j - Graph Database)

    ```cypher
    // Node: Intersection
    CREATE (i:Intersection {
        id: 'int_123',
        lat: 40.7580,
        lng: -73.9855,
        name: 'Broadway & 42nd St',
        city_id: 'nyc',
        signal_count: 4,
        has_cameras: true
    })

    // Node: Road Segment
    CREATE (r:RoadSegment {
        id: 'road_123',
        name: 'Broadway',
        length_meters: 500,
        lanes: 4,
        speed_limit_kmh: 40,
        road_type: 'arterial'
    })

    // Relationship: Connection between intersections
    CREATE (i1:Intersection)-[c:CONNECTS_TO {
        road_segment_id: 'road_123',
        travel_time_seconds: 120,
        current_speed_kmh: 25,
        congestion_level: 'moderate',
        last_updated: datetime()
    }]->(i2:Intersection)

    // Query: Find shortest path considering current traffic
    MATCH path = shortestPath(
        (start:Intersection {id: 'int_123'})-[c:CONNECTS_TO*]->(end:Intersection {id: 'int_789'})
    )
    WHERE ALL(rel IN relationships(path) WHERE rel.current_speed_kmh > 15)
    RETURN path,
           reduce(time = 0, rel IN relationships(path) | time + rel.travel_time_seconds) AS total_time
    ORDER BY total_time
    LIMIT 3;
    ```

    ---

    ### Incidents (PostgreSQL)

    ```sql
    -- Incidents table
    CREATE TABLE incidents (
        incident_id UUID PRIMARY KEY,
        type VARCHAR(50) NOT NULL,  -- accident, breakdown, road_hazard, construction
        severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
        status VARCHAR(20) NOT NULL,  -- reported, confirmed, in_progress, resolved
        location POINT NOT NULL,
        road_id VARCHAR(50),
        intersection_id VARCHAR(50),
        description TEXT,
        reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confirmed_at TIMESTAMP,
        resolved_at TIMESTAMP,
        reporter_id UUID,
        affected_lanes INT,
        estimated_clearance_time TIMESTAMP,
        INDEX idx_location USING GIST(location),
        INDEX idx_status_time (status, reported_at),
        INDEX idx_road (road_id, reported_at)
    );

    -- Signal configurations
    CREATE TABLE signal_configs (
        intersection_id VARCHAR(50) PRIMARY KEY,
        city_id VARCHAR(20) NOT NULL,
        current_timing JSONB NOT NULL,  -- {north_south_green: 45, east_west_green: 35, ...}
        optimization_mode VARCHAR(20),  -- manual, adaptive, ml_optimized
        last_optimized_at TIMESTAMP,
        performance_metrics JSONB,  -- {avg_delay: 35, throughput: 1200, ...}
        INDEX idx_city (city_id),
        INDEX idx_optimization (optimization_mode, last_optimized_at)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Real-time Traffic Signal Optimization Flow

    ```mermaid
    sequenceDiagram
        participant Sensor as Traffic Sensor
        participant Edge as Edge Node
        participant Stream as Stream Processor
        participant Analyzer as Traffic Analyzer
        participant ML as ML Optimizer
        participant Signal as Traffic Signal

        Sensor->>Edge: Vehicle detected (every 5s)
        Edge->>Edge: Aggregate local data
        Edge->>Stream: Send batch (every 30s)

        Stream->>Analyzer: Process sensor data
        Analyzer->>Analyzer: Calculate flow metrics
        Analyzer->>Redis: Update real-time state

        alt Congestion detected
            Analyzer->>ML: Request optimization
            ML->>ML: Run RL model (PPO)
            ML-->>Analyzer: Optimized timing plan
            Analyzer->>Edge: Send new timing
            Edge->>Signal: Adjust signal timing
            Signal-->>Edge: Acknowledgment
        else Normal flow
            Analyzer->>Analyzer: Monitor only
        end

        Note over Sensor,Signal: Total latency: 30-40 seconds for optimization
    ```

    ### Incident Detection and Response Flow

    ```mermaid
    sequenceDiagram
        participant Camera as Traffic Camera
        participant CV as Computer Vision
        participant Detector as Incident Detector
        participant Emergency as Emergency Mgmt
        participant Routing as Routing Service
        participant User as Navigation Users

        Camera->>CV: Video stream
        CV->>CV: Detect anomaly (stopped vehicles)
        CV->>Detector: Potential incident

        Detector->>Detector: Classify incident type
        Detector->>DB: Store incident

        alt High severity (accident)
            Detector->>Emergency: Alert emergency services
            Emergency->>Signal: Request green wave
            Signal-->>Emergency: Path cleared
        end

        Detector->>Routing: Update road status
        Routing->>Routing: Recalculate affected routes
        Routing->>User: Push notification (reroute)

        Note over Camera,User: Detection to notification: < 2 minutes
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical smart traffic subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Traffic Flow Models** | How to analyze traffic patterns? | Macroscopic flow models + ML |
    | **Signal Optimization** | How to optimize multi-intersection timing? | Reinforcement learning (PPO) |
    | **Congestion Prediction** | How to predict traffic 30-60 min ahead? | LSTM with spatial-temporal features |
    | **Incident Detection** | How to detect accidents in real-time? | Computer vision (YOLO) + anomaly detection |

    ---

    === "🚗 Traffic Flow Analysis"

        ## The Challenge

        **Problem:** Analyze traffic flow across 10K intersections with 1M sensor updates/sec.

        **Requirements:**
        - Real-time metrics: speed, density, volume, occupancy
        - Identify congestion patterns
        - Feed data to prediction and optimization models

        ---

        ## Traffic Flow Theory

        **Fundamental equation:**

        ```
        Flow (q) = Density (k) × Speed (v)

        q = vehicles/hour
        k = vehicles/km
        v = km/hour
        ```

        **Traffic states:**

        | State | Density | Speed | Flow | Description |
        |-------|---------|-------|------|-------------|
        | **Free Flow** | Low (< 20 veh/km) | High (50+ km/h) | Moderate | Vehicles move freely |
        | **Synchronized** | Medium (20-50 veh/km) | Medium (30-50 km/h) | High | Dense but moving |
        | **Congested** | High (> 50 veh/km) | Low (< 30 km/h) | Low | Stop-and-go |
        | **Gridlock** | Very high | Near zero | Near zero | Complete standstill |

        ---

        ## Implementation

        ```python
        import numpy as np
        from dataclasses import dataclass
        from typing import List, Dict
        import time

        @dataclass
        class TrafficMetrics:
            """Traffic flow metrics for a road segment"""
            vehicle_count: int
            average_speed: float  # km/h
            density: float  # vehicles/km
            flow: float  # vehicles/hour
            occupancy: float  # percentage
            congestion_level: str  # free, synchronized, congested, gridlock

        class TrafficFlowAnalyzer:
            """
            Analyze traffic flow using sensor data

            Implements macroscopic traffic flow theory
            """

            # Congestion thresholds
            DENSITY_THRESHOLDS = {
                'free': 20,          # vehicles/km
                'synchronized': 50,
                'congested': 80
            }

            SPEED_THRESHOLDS = {
                'free': 40,          # km/h
                'synchronized': 25,
                'congested': 15
            }

            def __init__(self):
                self.sensor_cache = {}  # Store recent sensor readings
                self.road_segments = {}  # Road segment metadata

            def analyze_segment(
                self,
                segment_id: str,
                sensor_readings: List[Dict],
                time_window_seconds: int = 300
            ) -> TrafficMetrics:
                """
                Analyze traffic flow for a road segment

                Args:
                    segment_id: Road segment identifier
                    sensor_readings: List of sensor data points
                    time_window_seconds: Analysis time window

                Returns:
                    TrafficMetrics with flow analysis
                """
                if not sensor_readings:
                    return self._default_metrics()

                # Get segment metadata
                segment = self.road_segments.get(segment_id, {})
                segment_length_km = segment.get('length_km', 1.0)
                num_lanes = segment.get('lanes', 2)

                # Calculate basic metrics
                vehicle_count = sum(r.get('vehicle_count', 0) for r in sensor_readings)
                speeds = [r.get('speed', 0) for r in sensor_readings if r.get('speed', 0) > 0]
                average_speed = np.mean(speeds) if speeds else 0

                # Calculate density (vehicles per km per lane)
                density = vehicle_count / (segment_length_km * num_lanes)

                # Calculate flow (vehicles per hour)
                # Flow = (vehicle_count / time_window) * 3600
                flow = (vehicle_count / time_window_seconds) * 3600

                # Calculate occupancy (percentage of time segment is occupied)
                occupancy = min(100, (density / 150) * 100)  # Normalized to jam density

                # Classify congestion level
                congestion_level = self._classify_congestion(density, average_speed)

                return TrafficMetrics(
                    vehicle_count=vehicle_count,
                    average_speed=average_speed,
                    density=density,
                    flow=flow,
                    occupancy=occupancy,
                    congestion_level=congestion_level
                )

            def _classify_congestion(self, density: float, speed: float) -> str:
                """
                Classify congestion level based on density and speed

                Uses both metrics for robust classification
                """
                # Gridlock: very high density AND very low speed
                if density > self.DENSITY_THRESHOLDS['congested'] and speed < 10:
                    return 'gridlock'

                # Congested: high density OR low speed
                if density > self.DENSITY_THRESHOLDS['congested'] or speed < self.SPEED_THRESHOLDS['congested']:
                    return 'congested'

                # Synchronized: medium density and medium speed
                if density > self.DENSITY_THRESHOLDS['synchronized'] or speed < self.SPEED_THRESHOLDS['synchronized']:
                    return 'synchronized'

                # Free flow
                return 'free'

            def detect_shockwave(
                self,
                segment_ids: List[str],
                current_metrics: Dict[str, TrafficMetrics]
            ) -> bool:
                """
                Detect traffic shockwave (congestion wave moving backward)

                Shockwave occurs when downstream congestion propagates upstream
                """
                if len(segment_ids) < 3:
                    return False

                # Check if congestion is propagating upstream
                congestion_levels = ['gridlock', 'congested', 'synchronized', 'free']

                for i in range(len(segment_ids) - 1):
                    upstream = segment_ids[i]
                    downstream = segment_ids[i + 1]

                    up_level = current_metrics.get(upstream, self._default_metrics()).congestion_level
                    down_level = current_metrics.get(downstream, self._default_metrics()).congestion_level

                    up_idx = congestion_levels.index(up_level) if up_level in congestion_levels else -1
                    down_idx = congestion_levels.index(down_level) if down_level in congestion_levels else -1

                    # Shockwave: upstream less congested than downstream
                    if up_idx > down_idx and up_idx <= 1:  # Upstream is congested/gridlock
                        return True

                return False

            def calculate_bottleneck_impact(
                self,
                intersection_id: str,
                approaching_segments: List[str],
                metrics: Dict[str, TrafficMetrics]
            ) -> float:
                """
                Calculate bottleneck severity at intersection

                Returns:
                    Impact score (0-1, higher = more severe bottleneck)
                """
                # Get metrics for all approaching segments
                segment_metrics = [metrics.get(seg_id) for seg_id in approaching_segments]
                segment_metrics = [m for m in segment_metrics if m is not None]

                if not segment_metrics:
                    return 0.0

                # Calculate average flow imbalance
                flows = [m.flow for m in segment_metrics]
                densities = [m.density for m in segment_metrics]

                # High variance in flows + high average density = bottleneck
                flow_variance = np.var(flows) if len(flows) > 1 else 0
                avg_density = np.mean(densities)

                # Normalize to 0-1 scale
                bottleneck_score = min(1.0, (flow_variance / 1000 + avg_density / 100) / 2)

                return bottleneck_score

            def _default_metrics(self) -> TrafficMetrics:
                """Return default metrics when no data available"""
                return TrafficMetrics(
                    vehicle_count=0,
                    average_speed=0,
                    density=0,
                    flow=0,
                    occupancy=0,
                    congestion_level='unknown'
                )
        ```

        **Performance:**
        - Process 1M sensor readings/sec with stream processing (Flink)
        - Calculate metrics in < 10ms per segment
        - Store aggregated results every 30 seconds

    === "🔄 Signal Optimization with RL"

        ## The Challenge

        **Problem:** Optimize traffic signal timing for 10K intersections to minimize total delay.

        **Traditional approaches:**
        - **Fixed timing:** Inflexible, doesn't adapt to traffic
        - **Actuated control:** Responds to sensors but local optimization only
        - **Coordinated timing:** Pre-programmed patterns for corridors

        **Our approach:** Reinforcement Learning (PPO) for adaptive optimization

        ---

        ## Reinforcement Learning Formulation

        **MDP (Markov Decision Process):**

        - **State (s):** Current traffic conditions
          - Vehicle counts on each approach
          - Queue lengths
          - Current signal phase
          - Time in current phase
          - Neighboring intersection states

        - **Action (a):** Signal timing decision
          - Extend current green (5s, 10s, 15s)
          - Switch to next phase
          - Emergency vehicle override

        - **Reward (r):** Traffic efficiency
          - Negative reward for vehicle delay
          - Positive reward for throughput
          - Penalty for phase switches (stability)

        ```
        Reward = -α × total_delay - β × queue_length + γ × throughput - δ × phase_switches

        where: α=1.0, β=0.5, γ=2.0, δ=0.1
        ```

        ---

        ## Implementation

        ```python
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from collections import deque
        import random

        class TrafficSignalEnvironment:
            """
            Traffic signal control environment for RL

            Implements OpenAI Gym-style interface
            """

            def __init__(self, intersection_id: str):
                self.intersection_id = intersection_id
                self.current_phase = 0  # 0: NS green, 1: EW green
                self.phase_time = 0
                self.total_delay = 0

                # State space: 4 approaches × 3 features + 2 signal state
                self.state_dim = 4 * 3 + 2  # [queue, speed, volume] × 4 + [phase, time]

                # Action space: 3 actions
                # 0: extend current phase 5s
                # 1: extend current phase 10s
                # 2: switch to next phase
                self.action_dim = 3

                # Traffic simulation parameters
                self.approach_queues = [0, 0, 0, 0]  # N, S, E, W
                self.approach_volumes = [0, 0, 0, 0]
                self.approach_speeds = [30.0, 30.0, 30.0, 30.0]

            def reset(self):
                """Reset environment to initial state"""
                self.current_phase = 0
                self.phase_time = 0
                self.total_delay = 0
                self.approach_queues = [5, 5, 5, 5]  # Initial queues
                self.approach_volumes = [20, 18, 15, 12]  # vehicles/min
                return self._get_state()

            def step(self, action: int):
                """
                Execute action and return new state

                Returns:
                    state, reward, done, info
                """
                # Execute action
                phase_switch = False

                if action == 0:  # Extend 5s
                    self.phase_time += 5
                elif action == 1:  # Extend 10s
                    self.phase_time += 10
                elif action == 2:  # Switch phase
                    self.current_phase = 1 - self.current_phase
                    self.phase_time = 0
                    phase_switch = True

                # Simulate traffic dynamics
                self._simulate_traffic_flow(5)  # 5-second time step

                # Calculate reward
                reward = self._calculate_reward(phase_switch)

                # Check if episode done (e.g., 1 hour simulation)
                done = self.total_delay > 3600  # 1 hour

                info = {
                    'total_delay': self.total_delay,
                    'queues': self.approach_queues.copy(),
                    'throughput': sum(self.approach_volumes)
                }

                return self._get_state(), reward, done, info

            def _get_state(self) -> np.ndarray:
                """Get current state representation"""
                state = []

                # Approach features
                for i in range(4):
                    state.extend([
                        self.approach_queues[i] / 50.0,  # Normalized queue
                        self.approach_speeds[i] / 60.0,   # Normalized speed
                        self.approach_volumes[i] / 30.0   # Normalized volume
                    ])

                # Signal state
                state.append(self.current_phase)
                state.append(self.phase_time / 90.0)  # Normalized time

                return np.array(state, dtype=np.float32)

            def _simulate_traffic_flow(self, time_step: int):
                """
                Simulate traffic flow for given time step

                Simple queue model:
                - Green phase: vehicles depart (discharge rate)
                - Red phase: vehicles accumulate (arrival rate)
                """
                discharge_rate = 0.5  # vehicles/second (green)

                for i in range(4):
                    # Arrivals (random with volume mean)
                    arrivals = np.random.poisson(self.approach_volumes[i] * time_step / 60.0)
                    self.approach_queues[i] += arrivals

                    # Departures (only if green)
                    if (i < 2 and self.current_phase == 0) or (i >= 2 and self.current_phase == 1):
                        # North-South (0,1) or East-West (2,3) green
                        departures = min(
                            self.approach_queues[i],
                            int(discharge_rate * time_step)
                        )
                        self.approach_queues[i] -= departures
                    else:
                        # Red phase: accumulate delay
                        self.total_delay += self.approach_queues[i] * time_step

                    # Update speed based on queue
                    if self.approach_queues[i] > 20:
                        self.approach_speeds[i] = max(10.0, 30.0 - self.approach_queues[i] * 0.5)
                    else:
                        self.approach_speeds[i] = 30.0

            def _calculate_reward(self, phase_switch: bool) -> float:
                """Calculate reward based on traffic efficiency"""
                # Penalize total delay
                delay_penalty = -sum(self.approach_queues) * 0.1

                # Reward throughput
                throughput_reward = sum(self.approach_volumes) * 0.05

                # Penalize phase switches (stability)
                switch_penalty = -5.0 if phase_switch else 0

                # Penalize long queues (safety)
                queue_penalty = -sum(max(0, q - 30) for q in self.approach_queues) * 0.2

                total_reward = delay_penalty + throughput_reward + switch_penalty + queue_penalty

                return total_reward

        class SignalOptimizer(nn.Module):
            """
            Neural network for signal optimization using PPO

            Actor-Critic architecture
            """

            def __init__(self, state_dim: int, action_dim: int):
                super().__init__()

                # Shared feature extractor
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )

                # Actor head (policy)
                self.actor = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                    nn.Softmax(dim=-1)
                )

                # Critic head (value function)
                self.critic = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

            def forward(self, state):
                """Forward pass through network"""
                features = self.shared(state)
                action_probs = self.actor(features)
                state_value = self.critic(features)
                return action_probs, state_value

            def select_action(self, state):
                """Select action using current policy"""
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, state_value = self.forward(state_tensor)

                # Sample action from probability distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                return action.item(), action_dist.log_prob(action), state_value

        class PPOTrainer:
            """
            Proximal Policy Optimization trainer

            Train signal optimization policy using PPO algorithm
            """

            def __init__(self, state_dim: int, action_dim: int):
                self.policy = SignalOptimizer(state_dim, action_dim)
                self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)

                # PPO hyperparameters
                self.clip_epsilon = 0.2
                self.gamma = 0.99
                self.gae_lambda = 0.95

                self.memory = []

            def train_episode(self, env: TrafficSignalEnvironment, num_steps: int = 1000):
                """Train policy for one episode"""
                state = env.reset()
                episode_reward = 0

                states, actions, rewards, log_probs, values = [], [], [], [], []

                for step in range(num_steps):
                    # Select action
                    action, log_prob, value = self.policy.select_action(state)

                    # Execute action
                    next_state, reward, done, info = env.step(action)

                    # Store transition
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    values.append(value)

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                # Update policy using collected experience
                self._update_policy(states, actions, rewards, log_probs, values)

                return episode_reward, info

            def _update_policy(self, states, actions, rewards, log_probs, values):
                """Update policy using PPO objective"""
                # Calculate returns and advantages using GAE
                returns = self._calculate_returns(rewards, values)
                advantages = returns - torch.cat(values).detach()

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.LongTensor(actions)
                old_log_probs = torch.cat(log_probs).detach()

                # PPO update (multiple epochs)
                for _ in range(10):
                    # Get current policy predictions
                    action_probs, state_values = self.policy(states_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = action_dist.log_prob(actions_tensor)

                    # Calculate PPO objective
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                    policy_loss = -torch.min(
                        ratio * advantages,
                        clipped_ratio * advantages
                    ).mean()

                    value_loss = nn.MSELoss()(state_values.squeeze(), returns)

                    # Total loss
                    loss = policy_loss + 0.5 * value_loss

                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            def _calculate_returns(self, rewards, values):
                """Calculate discounted returns using GAE"""
                returns = []
                gae = 0

                for i in reversed(range(len(rewards))):
                    if i == len(rewards) - 1:
                        next_value = 0
                    else:
                        next_value = values[i + 1]

                    delta = rewards[i] + self.gamma * next_value - values[i]
                    gae = delta + self.gamma * self.gae_lambda * gae
                    returns.insert(0, gae + values[i])

                return torch.FloatTensor(returns)

        # Example usage
        def train_signal_optimizer():
            """Train signal optimization policy"""
            env = TrafficSignalEnvironment('int_123')
            trainer = PPOTrainer(state_dim=env.state_dim, action_dim=env.action_dim)

            num_episodes = 1000
            for episode in range(num_episodes):
                episode_reward, info = trainer.train_episode(env)

                if episode % 100 == 0:
                    print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Total Delay={info['total_delay']:.0f}s, "
                          f"Avg Queue={np.mean(info['queues']):.1f}")

            return trainer.policy
        ```

        **Real-world deployment:**
        - Train on simulation with historical traffic data
        - Fine-tune on real intersection with A/B testing
        - Deploy to edge nodes for real-time control
        - Monitor performance and retrain monthly

    === "📈 Congestion Prediction with LSTM"

        ## The Challenge

        **Problem:** Predict traffic congestion 30-60 minutes ahead with > 85% accuracy.

        **Why it's hard:**
        - Temporal dependencies (rush hour patterns)
        - Spatial dependencies (congestion spreads)
        - External factors (weather, events, incidents)

        **Solution:** LSTM (Long Short-Term Memory) with spatial-temporal features

        ---

        ## Model Architecture

        ```python
        import torch
        import torch.nn as nn
        import numpy as np
        from typing import List, Tuple

        class SpatialTemporalLSTM(nn.Module):
            """
            LSTM model for traffic congestion prediction

            Combines spatial features (road network) with temporal patterns
            """

            def __init__(
                self,
                num_features: int = 10,
                hidden_size: int = 128,
                num_layers: int = 2,
                num_road_segments: int = 100,
                prediction_horizon: int = 6  # 6 × 10min = 1 hour
            ):
                super().__init__()

                self.num_features = num_features
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.prediction_horizon = prediction_horizon

                # Spatial feature encoder (per road segment)
                self.spatial_encoder = nn.Sequential(
                    nn.Linear(num_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )

                # Graph attention for spatial dependencies
                self.graph_attention = nn.MultiheadAttention(
                    embed_dim=32,
                    num_heads=4,
                    batch_first=True
                )

                # Temporal LSTM
                self.lstm = nn.LSTM(
                    input_size=32,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2
                )

                # Prediction head
                self.predictor = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, prediction_horizon * 4)  # 4 classes: free, moderate, heavy, severe
                )

                # Output activation
                self.softmax = nn.Softmax(dim=-1)

            def forward(
                self,
                x: torch.Tensor,
                adjacency_matrix: torch.Tensor = None
            ) -> torch.Tensor:
                """
                Forward pass

                Args:
                    x: Input tensor [batch, sequence_length, num_segments, num_features]
                    adjacency_matrix: Spatial connectivity [num_segments, num_segments]

                Returns:
                    Predictions [batch, prediction_horizon, num_classes]
                """
                batch_size, seq_len, num_segments, num_features = x.shape

                # Encode spatial features for each time step
                spatial_features = []
                for t in range(seq_len):
                    # Encode features for all segments
                    segment_features = self.spatial_encoder(x[:, t, :, :])  # [batch, num_segments, 32]

                    # Apply graph attention to capture spatial dependencies
                    attended_features, _ = self.graph_attention(
                        segment_features,
                        segment_features,
                        segment_features
                    )

                    # Aggregate across segments (mean pooling)
                    aggregated = torch.mean(attended_features, dim=1)  # [batch, 32]
                    spatial_features.append(aggregated)

                # Stack temporal sequence
                temporal_input = torch.stack(spatial_features, dim=1)  # [batch, seq_len, 32]

                # LSTM for temporal patterns
                lstm_out, _ = self.lstm(temporal_input)  # [batch, seq_len, hidden_size]

                # Use last hidden state for prediction
                last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]

                # Predict future congestion
                predictions = self.predictor(last_hidden)  # [batch, horizon × 4]

                # Reshape and apply softmax
                predictions = predictions.view(batch_size, self.prediction_horizon, 4)
                predictions = self.softmax(predictions)

                return predictions

        class TrafficPredictor:
            """
            Traffic congestion prediction service

            Manages model inference and feature engineering
            """

            def __init__(self, model_path: str = None):
                self.model = SpatialTemporalLSTM()
                if model_path:
                    self.model.load_state_dict(torch.load(model_path))
                self.model.eval()

                # Feature statistics for normalization
                self.feature_stats = {
                    'speed': {'mean': 30.0, 'std': 15.0},
                    'volume': {'mean': 500.0, 'std': 200.0},
                    'density': {'mean': 40.0, 'std': 20.0},
                    'occupancy': {'mean': 30.0, 'std': 15.0}
                }

            def predict(
                self,
                historical_data: List[Dict],
                road_network: Dict,
                current_time: str,
                prediction_horizon_minutes: int = 60
            ) -> List[Dict]:
                """
                Predict traffic congestion

                Args:
                    historical_data: Past traffic data (last 2 hours)
                    road_network: Road network graph
                    current_time: Current timestamp
                    prediction_horizon_minutes: How far to predict

                Returns:
                    List of predictions with timestamps and confidence
                """
                # Prepare input features
                features = self._prepare_features(historical_data, current_time)

                # Add spatial context
                adjacency = self._build_adjacency_matrix(road_network)

                # Convert to tensors
                x = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
                adj = torch.FloatTensor(adjacency)

                # Run inference
                with torch.no_grad():
                    predictions = self.model(x, adj)

                # Convert to readable format
                results = self._format_predictions(
                    predictions.squeeze(0),
                    current_time,
                    prediction_horizon_minutes
                )

                return results

            def _prepare_features(
                self,
                historical_data: List[Dict],
                current_time: str
            ) -> np.ndarray:
                """
                Prepare input features from historical data

                Features per segment:
                - Average speed
                - Vehicle volume
                - Density
                - Occupancy
                - Hour of day (sin/cos encoded)
                - Day of week (sin/cos encoded)
                - Is weekend (binary)
                - Is rush hour (binary)
                - Weather condition (categorical → one-hot)
                """
                # Parse time features
                from datetime import datetime
                dt = datetime.fromisoformat(current_time)
                hour = dt.hour
                day_of_week = dt.weekday()

                # Temporal features (cyclic encoding)
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                day_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_cos = np.cos(2 * np.pi * day_of_week / 7)
                is_weekend = 1 if day_of_week >= 5 else 0
                is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0

                # Process historical data (last 12 time steps = 2 hours)
                sequence_length = 12
                num_segments = len(set(d['segment_id'] for d in historical_data))
                num_features = 10

                features = np.zeros((sequence_length, num_segments, num_features))

                # Group by time step and segment
                for t_idx in range(sequence_length):
                    for seg_idx, segment_data in enumerate(historical_data):
                        if segment_data['time_step'] == t_idx:
                            # Traffic features (normalized)
                            speed = (segment_data['speed'] - self.feature_stats['speed']['mean']) / self.feature_stats['speed']['std']
                            volume = (segment_data['volume'] - self.feature_stats['volume']['mean']) / self.feature_stats['volume']['std']
                            density = (segment_data['density'] - self.feature_stats['density']['mean']) / self.feature_stats['density']['std']
                            occupancy = (segment_data['occupancy'] - self.feature_stats['occupancy']['mean']) / self.feature_stats['occupancy']['std']

                            # Combine features
                            features[t_idx, seg_idx, :] = [
                                speed,
                                volume,
                                density,
                                occupancy,
                                hour_sin,
                                hour_cos,
                                day_sin,
                                day_cos,
                                is_weekend,
                                is_rush_hour
                            ]

                return features

            def _build_adjacency_matrix(self, road_network: Dict) -> np.ndarray:
                """Build spatial adjacency matrix from road network"""
                num_segments = len(road_network['segments'])
                adjacency = np.zeros((num_segments, num_segments))

                # Fill adjacency based on connections
                for connection in road_network['connections']:
                    from_idx = connection['from']
                    to_idx = connection['to']
                    adjacency[from_idx, to_idx] = 1
                    adjacency[to_idx, from_idx] = 1  # Undirected

                # Add self-connections
                np.fill_diagonal(adjacency, 1)

                return adjacency

            def _format_predictions(
                self,
                predictions: torch.Tensor,
                current_time: str,
                horizon_minutes: int
            ) -> List[Dict]:
                """Format model predictions into readable output"""
                from datetime import datetime, timedelta

                dt = datetime.fromisoformat(current_time)
                results = []

                # Congestion classes
                classes = ['free', 'moderate', 'heavy', 'severe']

                for t_idx in range(predictions.shape[0]):
                    # Future timestamp
                    future_time = dt + timedelta(minutes=(t_idx + 1) * 10)

                    # Get class probabilities
                    probs = predictions[t_idx].numpy()
                    predicted_class = classes[np.argmax(probs)]
                    confidence = float(np.max(probs))

                    results.append({
                        'timestamp': future_time.isoformat(),
                        'congestion_level': predicted_class,
                        'confidence': confidence,
                        'probabilities': {
                            cls: float(prob) for cls, prob in zip(classes, probs)
                        }
                    })

                return results

        # Training function
        def train_prediction_model(
            train_data: List[Tuple[np.ndarray, np.ndarray]],
            val_data: List[Tuple[np.ndarray, np.ndarray]],
            num_epochs: int = 100
        ):
            """
            Train congestion prediction model

            Args:
                train_data: List of (features, labels) tuples
                val_data: Validation data
                num_epochs: Number of training epochs
            """
            model = SpatialTemporalLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            best_val_accuracy = 0

            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for features, labels in train_data:
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(torch.FloatTensor(features))

                    # Flatten for loss calculation
                    pred_flat = predictions.view(-1, 4)
                    label_flat = labels.view(-1)

                    # Calculate loss
                    loss = criterion(pred_flat, label_flat)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(pred_flat, 1)
                    train_correct += (predicted == label_flat).sum().item()
                    train_total += label_flat.size(0)

                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for features, labels in val_data:
                        predictions = model(torch.FloatTensor(features))
                        pred_flat = predictions.view(-1, 4)
                        label_flat = labels.view(-1)

                        loss = criterion(pred_flat, label_flat)
                        val_loss += loss.item()

                        _, predicted = torch.max(pred_flat, 1)
                        val_correct += (predicted == label_flat).sum().item()
                        val_total += label_flat.size(0)

                train_accuracy = 100 * train_correct / train_total
                val_accuracy = 100 * val_correct / val_total

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'best_model.pth')

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%")

            return model
        ```

        **Performance:**
        - Prediction accuracy: 87% for 30-minute ahead, 82% for 60-minute ahead
        - Inference time: < 50ms per prediction
        - Update model daily with new traffic patterns

    === "🚨 Incident Detection with Computer Vision"

        ## The Challenge

        **Problem:** Detect traffic incidents (accidents, breakdowns) in real-time from 100K cameras.

        **Requirements:**
        - Detection latency: < 2 minutes
        - Accuracy: > 90% detection rate
        - Low false positive rate: < 5%

        **Solution:** YOLO (You Only Look Once) + anomaly detection

        ---

        ## Implementation

        ```python
        import cv2
        import numpy as np
        from typing import List, Dict, Tuple
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from collections import deque

        class IncidentDetector:
            """
            Real-time incident detection from traffic cameras

            Uses computer vision to detect:
            - Stopped vehicles
            - Accidents
            - Wrong-way drivers
            - Pedestrians on highway
            """

            def __init__(self, model_path: str = None):
                # Load object detection model (YOLO or Faster R-CNN)
                self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
                self.detector.eval()

                # Vehicle tracking
                self.vehicle_tracks = {}  # vehicle_id -> [positions]
                self.stopped_vehicles = {}  # vehicle_id -> stopped_duration

                # Anomaly detection thresholds
                self.STOPPED_THRESHOLD = 60  # seconds
                self.SPEED_THRESHOLD = 5  # km/h

            def process_frame(
                self,
                frame: np.ndarray,
                camera_id: str,
                timestamp: float
            ) -> List[Dict]:
                """
                Process single camera frame

                Args:
                    frame: Camera frame (BGR image)
                    camera_id: Camera identifier
                    timestamp: Frame timestamp

                Returns:
                    List of detected incidents
                """
                # Detect vehicles
                detections = self._detect_objects(frame)

                # Track vehicles
                tracks = self._update_tracks(detections, timestamp)

                # Detect anomalies
                incidents = []

                # 1. Stopped vehicles
                stopped = self._detect_stopped_vehicles(tracks, timestamp)
                incidents.extend(stopped)

                # 2. Wrong-way drivers
                wrong_way = self._detect_wrong_way(tracks)
                incidents.extend(wrong_way)

                # 3. Crowd formation (accident)
                crowd = self._detect_crowd(tracks)
                incidents.extend(crowd)

                return incidents

            def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
                """
                Detect objects in frame using deep learning

                Returns:
                    List of detections with bounding boxes
                """
                # Preprocess frame
                img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)

                # Run detection
                with torch.no_grad():
                    predictions = self.detector(img_tensor)

                # Extract vehicle detections
                detections = []
                pred = predictions[0]

                for i in range(len(pred['boxes'])):
                    score = pred['scores'][i].item()
                    if score < 0.7:
                        continue

                    box = pred['boxes'][i].cpu().numpy()
                    label = pred['labels'][i].item()

                    # Filter for vehicles (car, truck, bus, motorcycle)
                    if label in [3, 6, 8, 4]:  # COCO dataset labels
                        detections.append({
                            'box': box,
                            'score': score,
                            'label': label,
                            'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        })

                return detections

            def _update_tracks(
                self,
                detections: List[Dict],
                timestamp: float
            ) -> Dict[int, Dict]:
                """
                Update vehicle tracks using simple IoU-based tracking

                Returns:
                    Dictionary of active tracks
                """
                # Match detections to existing tracks
                matched_tracks = {}
                unmatched_detections = detections.copy()

                for track_id, track in self.vehicle_tracks.items():
                    best_match = None
                    best_iou = 0.3  # Minimum IoU threshold

                    for det in unmatched_detections:
                        iou = self._calculate_iou(track['last_box'], det['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = det

                    if best_match:
                        # Update track
                        track['positions'].append(best_match['center'])
                        track['last_box'] = best_match['box']
                        track['last_seen'] = timestamp
                        matched_tracks[track_id] = track
                        unmatched_detections.remove(best_match)

                # Create new tracks for unmatched detections
                next_id = max(self.vehicle_tracks.keys()) + 1 if self.vehicle_tracks else 0
                for det in unmatched_detections:
                    matched_tracks[next_id] = {
                        'positions': deque([det['center']], maxlen=30),
                        'last_box': det['box'],
                        'first_seen': timestamp,
                        'last_seen': timestamp
                    }
                    next_id += 1

                # Update global tracks
                self.vehicle_tracks = matched_tracks

                return matched_tracks

            def _detect_stopped_vehicles(
                self,
                tracks: Dict[int, Dict],
                timestamp: float
            ) -> List[Dict]:
                """
                Detect vehicles that have been stopped for too long
                """
                incidents = []

                for track_id, track in tracks.items():
                    # Check if vehicle has been stationary
                    if len(track['positions']) < 10:
                        continue

                    # Calculate movement in last 30 frames
                    recent_positions = list(track['positions'])[-30:]
                    movement = np.std([p[0] for p in recent_positions]) + np.std([p[1] for p in recent_positions])

                    # If movement is very small and duration is long
                    duration = timestamp - track['first_seen']
                    if movement < 5.0 and duration > self.STOPPED_THRESHOLD:
                        # Check if not already reported
                        if track_id not in self.stopped_vehicles:
                            self.stopped_vehicles[track_id] = timestamp

                            incidents.append({
                                'type': 'stopped_vehicle',
                                'severity': 'medium',
                                'location': recent_positions[-1],
                                'duration': duration,
                                'confidence': 0.85,
                                'timestamp': timestamp
                            })

                return incidents

            def _detect_wrong_way(self, tracks: Dict[int, Dict]) -> List[Dict]:
                """Detect vehicles moving in wrong direction"""
                incidents = []

                # Assume road direction is left-to-right (positive x)
                expected_direction = 1  # 1 for right, -1 for left

                for track_id, track in tracks.items():
                    if len(track['positions']) < 15:
                        continue

                    # Calculate direction of movement
                    positions = list(track['positions'])
                    dx = positions[-1][0] - positions[0][0]

                    # If moving opposite to expected direction
                    if dx * expected_direction < -50:  # Significant opposite movement
                        incidents.append({
                            'type': 'wrong_way_driver',
                            'severity': 'high',
                            'location': positions[-1],
                            'confidence': 0.90,
                            'timestamp': track['last_seen']
                        })

                return incidents

            def _detect_crowd(self, tracks: Dict[int, Dict]) -> List[Dict]:
                """Detect crowd of stopped vehicles (likely accident)"""
                incidents = []

                # Find clusters of stopped vehicles
                stopped_positions = []
                for track_id, track in tracks.items():
                    if len(track['positions']) < 5:
                        continue

                    # Check if stopped
                    recent = list(track['positions'])[-5:]
                    movement = np.std([p[0] for p in recent]) + np.std([p[1] for p in recent])

                    if movement < 3.0:
                        stopped_positions.append(recent[-1])

                # Cluster stopped vehicles
                if len(stopped_positions) >= 3:
                    # Simple clustering: check if vehicles are close
                    clusters = self._cluster_positions(stopped_positions, threshold=100)

                    for cluster in clusters:
                        if len(cluster) >= 3:
                            # Likely accident
                            center = np.mean(cluster, axis=0)
                            incidents.append({
                                'type': 'accident',
                                'severity': 'critical',
                                'location': tuple(center),
                                'affected_vehicles': len(cluster),
                                'confidence': 0.75,
                                'timestamp': time.time()
                            })

                return incidents

            def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
                """Calculate Intersection over Union for two bounding boxes"""
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)

                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0

            def _cluster_positions(
                self,
                positions: List[Tuple[float, float]],
                threshold: float = 100
            ) -> List[List[Tuple[float, float]]]:
                """Simple distance-based clustering"""
                if not positions:
                    return []

                clusters = []
                remaining = positions.copy()

                while remaining:
                    # Start new cluster with first position
                    current_cluster = [remaining.pop(0)]

                    # Add nearby positions
                    i = 0
                    while i < len(remaining):
                        pos = remaining[i]

                        # Check distance to any position in cluster
                        for cluster_pos in current_cluster:
                            dist = np.sqrt((pos[0] - cluster_pos[0])**2 + (pos[1] - cluster_pos[1])**2)
                            if dist < threshold:
                                current_cluster.append(pos)
                                remaining.pop(i)
                                break
                        else:
                            i += 1

                    clusters.append(current_cluster)

                return clusters
        ```

        **Real-world considerations:**
        - Deploy on edge servers near cameras
        - Process video at 5-10 FPS (not every frame)
        - Use motion detection to skip empty frames
        - Combine with other sensors (loop detectors, radar) for validation

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 1 city (10K intersections) to 50 cities (500K intersections).

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Sensor ingestion** | ✅ Yes | Kafka with 50 partitions, Flink for processing |
    | **Time-series storage** | ✅ Yes | InfluxDB cluster (20 nodes), retention policy (90 days) |
    | **ML inference** | 🟡 Moderate | GPU servers for prediction, edge nodes for signal control |
    | **Graph queries** | 🟡 Moderate | Neo4j cluster (10 nodes), cache hot paths in Redis |
    | **Camera processing** | ✅ Yes | Edge computing at intersections, upload incidents only |

    ---

    ## Edge Computing Architecture

    **Problem:** Cannot send all camera streams to cloud (bandwidth, latency).

    **Solution:** Process at edge, send aggregated data.

    ```
    Intersection Edge Node:
    ├── Video processing (incident detection)
    ├── Sensor aggregation (5-second batches)
    ├── Local signal control (< 100ms latency)
    └── Upstream communication (every 30s)

    Edge → Zone Controller → City Controller → Multi-City Controller
    ```

    **Benefits:**
    - 100x bandwidth reduction
    - Sub-100ms signal control latency
    - Resilient to network failures

    ---

    ## Data Aggregation Strategy

    **Raw sensor data:** 1M updates/sec = 17.3 TB/day

    **Aggregation levels:**

    1. **5-second aggregation:** Average metrics per segment
       - Reduce from 1M/sec to 200K/sec (5x)

    2. **1-minute aggregation:** For analytics
       - Reduce to 33K/sec (30x)

    3. **5-minute aggregation:** For historical analysis
       - Reduce to 6.6K/sec (150x)

    **Storage:**
    - Raw (5-sec): 90 days = 1.56 PB
    - 1-min: 1 year = 104 TB
    - 5-min: 5 years = 95 TB
    - Total: 1.76 PB

    ---

    ## Real-time vs Batch Processing

    **Real-time (Kafka + Flink):**
    - Signal optimization (immediate)
    - Incident detection (< 2 min)
    - Congestion alerts (< 5 min)

    **Batch (Spark):**
    - ML model training (daily)
    - Historical analytics (hourly)
    - Performance reports (weekly)

    ---

    ## Multi-City Coordination

    **Challenge:** Coordinate traffic across city boundaries.

    **Solution:** Hierarchical control

    ```
    Multi-City Controller
    ├── City A Controller (NYC)
    │   ├── Zone 1 (Manhattan)
    │   ├── Zone 2 (Brooklyn)
    │   └── Zone 3 (Queens)
    ├── City B Controller (Boston)
    └── City C Controller (Philadelphia)
    ```

    **Coordination scenarios:**
    - Major events (concerts, sports)
    - Regional incidents (bridge closure)
    - Inter-city highways (I-95 corridor)

    ---

    ## Cost Optimization

    **Monthly cost for 50 cities:**

    | Component | Cost |
    |-----------|------|
    | **Edge Nodes** | $500,000 (500K intersections × $100/month) |
    | **Kafka/Flink** | $50,000 (50 servers) |
    | **InfluxDB** | $80,000 (20 nodes) |
    | **Neo4j** | $40,000 (10 nodes) |
    | **ML Training** | $100,000 (GPU clusters) |
    | **Object Storage** | $200,000 (camera footage) |
    | **Networking** | $150,000 |
    | **Total** | **$1.12M/month** |

    **Cost per city:** $22,400/month

    **City benefits:**
    - 20% reduction in congestion
    - 30% faster emergency response
    - 15% reduction in emissions
    - ROI: Positive within 2 years

    ---

    ## Monitoring and Alerting

    **Key metrics:**

    1. **System health:**
       - Sensor uptime: > 98%
       - Signal response time: < 100ms
       - Prediction accuracy: > 85%

    2. **Traffic performance:**
       - Average delay per vehicle
       - Intersection throughput
       - Incident detection time
       - Green wave efficiency

    3. **ML model performance:**
       - Prediction accuracy (hourly)
       - False positive rate
       - Model drift detection

    **Alerting:**
    - PagerDuty for critical incidents
    - Slack for warnings
    - Email for daily reports

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Edge computing** - Process sensor data locally for < 100ms latency
    2. **Reinforcement learning** - Adaptive signal optimization
    3. **LSTM predictions** - Capture temporal patterns for congestion
    4. **Computer vision** - Real-time incident detection from cameras
    5. **Hierarchical control** - Intersection → Zone → City coordination
    6. **Data aggregation** - Reduce storage from 17TB/day to manageable levels

    ---

    ## Interview Tips

    ✅ **Start with requirements** - Clarify scale (city size, sensor count)

    ✅ **Discuss edge computing** - Why process at intersection vs cloud

    ✅ **ML models** - LSTM for prediction, RL for optimization, CV for detection

    ✅ **Data pipeline** - Kafka/Flink for streaming, InfluxDB for time-series

    ✅ **Trade-offs** - Real-time accuracy vs cost, edge vs cloud processing

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle sensor failures?"** | Redundant sensors, interpolation from neighbors, graceful degradation |
    | **"What if edge node loses connectivity?"** | Local autonomy, cached ML models, queue data for sync |
    | **"How to prevent signal timing oscillation?"** | Hysteresis, minimum phase duration, stability constraints in RL |
    | **"How to prioritize emergency vehicles?"** | GPS tracking, preemption signals, clear path optimization |
    | **"How to measure system effectiveness?"** | A/B testing zones, before/after analysis, control groups |

    ---

    ## Real-World Implementations

    **Reference systems:**

    1. **Los Angeles ATSAC** (Automated Traffic Surveillance and Control)
       - 4,500 intersections
       - Adaptive signal timing
       - 12% reduction in travel time

    2. **Singapore Green Link Determining System**
       - ML-based prediction
       - 20% improvement in traffic flow
       - Integration with public transit

    3. **Barcelona Smart City**
       - IoT sensors + edge computing
       - 21% less congestion
       - Real-time parking guidance

    4. **IBM Smarter Traffic (Singapore, Brisbane)**
       - Predictive analytics
       - Incident detection
       - Multi-modal coordination

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Siemens Mobility, IBM, Cisco, GE, City governments

---

*Master this problem and you'll be ready for: Smart city systems, IoT platforms, traffic management, urban computing*
