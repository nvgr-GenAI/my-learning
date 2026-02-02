# Design Food Delivery (UberEats/DoorDash)

A real-time food delivery platform where customers order from restaurants, drivers pick up and deliver orders, with real-time tracking, dynamic pricing, and intelligent matching algorithms.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M orders/day, 5M drivers, 1M restaurants, 100M users |
| **Key Challenges** | Real-time driver matching, route optimization, ETA prediction, surge pricing, concurrent order handling |
| **Core Concepts** | Geohashing, dispatch algorithms, dynamic pricing, real-time location tracking, GraphHopper routing |
| **Companies** | UberEats, DoorDash, GrubHub, Postmates, Deliveroo, Swiggy, Zomato |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Search Restaurants** | Browse nearby restaurants by cuisine, rating, delivery time | P0 (Must have) |
    | **Place Order** | Add items to cart, checkout, payment | P0 (Must have) |
    | **Driver Matching** | Automatically match available driver to order | P0 (Must have) |
    | **Real-time Tracking** | Track driver location, ETA updates | P0 (Must have) |
    | **Order Status** | Order preparation, pickup, delivery status updates | P0 (Must have) |
    | **Payment Processing** | Credit/debit cards, digital wallets | P0 (Must have) |
    | **Driver Navigation** | Turn-by-turn directions, route optimization | P1 (Should have) |
    | **Ratings & Reviews** | Rate restaurant, food quality, driver | P1 (Should have) |
    | **Surge Pricing** | Dynamic pricing based on demand | P1 (Should have) |
    | **Delivery Zones** | Define service areas, delivery radius | P1 (Should have) |
    | **Schedule Orders** | Pre-order for later delivery | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Restaurant inventory management
    - Kitchen display systems
    - Customer support chat
    - Loyalty programs
    - Advertising platform
    - Fraud detection (ML models)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Critical during meal times (lunch/dinner rush) |
    | **Latency (Search)** | < 500ms p95 | Fast restaurant browsing essential for UX |
    | **Latency (Matching)** | < 30s | Quick driver assignment critical for food quality |
    | **ETA Accuracy** | < 5min error | Users plan around delivery time |
    | **Location Update Frequency** | Every 5-10s | Smooth real-time tracking experience |
    | **Scalability** | 100K concurrent orders | Handle dinner rush in major cities |
    | **Consistency** | Eventual consistency | Brief delays acceptable (status updates may lag) |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 20M (20% of 100M users)
    Monthly Active Users (MAU): 100M

    Order creation:
    - Orders per DAU: ~0.5 orders/day (not everyone orders daily)
    - Daily orders: 20M × 2.5 (avg) = 50M orders/day
    - Order QPS: 50M / 86,400 = ~580 orders/sec
    - Peak QPS: 5x average = ~2,900 orders/sec (dinner rush 6-8pm)

    Restaurant browsing:
    - Browse sessions per DAU: ~5 sessions/day (window shopping)
    - Daily search requests: 20M × 5 = 100M requests/day
    - Search QPS: 100M / 86,400 = ~1,160 req/sec

    Location updates (drivers):
    - Active drivers during peak: 2M (40% of 5M)
    - Update frequency: Every 5 seconds
    - Location update QPS: 2M / 5 = 400K updates/sec

    Status updates (orders):
    - Active orders: 2M concurrent (peak)
    - Status changes per order: ~8 (placed, accepted, preparing, ready, picked, transit, nearby, delivered)
    - Duration: ~45 min average
    - Status update QPS: (2M × 8) / 2,700 = ~6K updates/sec

    Total Read QPS: ~50K (search + tracking + restaurant data)
    Total Write QPS: ~410K (location updates + status + orders)
    Read/Write ratio: 1:8 (write-heavy due to location tracking)
    ```

    ### Storage Estimates

    ```
    Order storage:
    - Order metadata: 5 KB (order_id, user_id, restaurant_id, items, total, status, timestamps)
    - Items: 2 KB (3 items × 700 bytes average)
    - Total per order: ~7 KB

    For 5 years:
    - Orders: 50M/day × 365 × 5 = 91.25 billion orders
    - Storage: 91.25B × 7 KB = 638 TB

    User data:
    - 100M users × 5 KB = 500 GB
    - Delivery addresses: 100M × 2 addresses × 500 bytes = 100 GB

    Restaurant data:
    - 1M restaurants × 100 KB (menu, images, metadata) = 100 GB

    Driver data:
    - 5M drivers × 10 KB = 50 GB

    Location history (hot data - last 7 days):
    - 2M active drivers × 12 updates/min × 60 min × 24 hours × 7 days × 100 bytes
    - = 2M × 120,960 × 100 bytes = 24 TB

    Total: 638 TB (orders) + 100 GB (users) + 100 GB (restaurants) + 24 TB (location) ≈ 662 TB
    ```

    ### Bandwidth Estimates

    ```
    Order ingress:
    - 580 orders/sec × 7 KB = 4 MB/sec ≈ 32 Mbps

    Location updates ingress:
    - 400K updates/sec × 100 bytes = 40 MB/sec ≈ 320 Mbps

    Restaurant search egress:
    - 1,160 searches/sec × 20 restaurants × 10 KB = 232 MB/sec ≈ 1.9 Gbps

    Location tracking egress (real-time):
    - 2M active orders × 1 update/5s × 100 bytes = 40 MB/sec ≈ 320 Mbps

    Total ingress: ~352 Mbps
    Total egress: ~2.2 Gbps (CDN for restaurant images critical)
    ```

    ### Memory Estimates (Caching)

    ```
    Restaurant data (hot cache):
    - 100K popular restaurants × 100 KB = 10 GB
    - Cache 80% hit rate: 8 GB effective

    Active orders:
    - 2M concurrent orders × 10 KB = 20 GB

    Driver locations (real-time):
    - 2M active drivers × 500 bytes (location, status, metadata) = 1 GB

    Menu cache:
    - 50K popular menus × 200 KB = 10 GB

    Geohash index:
    - 10M location cells × 1 KB = 10 GB

    Route cache (ETA calculations):
    - 1M cached routes × 5 KB = 5 GB

    Total cache: 8 GB + 20 GB + 1 GB + 10 GB + 10 GB + 5 GB ≈ 54 GB
    ```

    ---

    ## Key Assumptions

    1. Average order value: $30
    2. Average delivery time: 45 minutes
    3. Average delivery radius: 5 miles
    4. Driver acceptance rate: 80%
    5. Peak hours: 11am-1pm (lunch), 6pm-8pm (dinner)
    6. 40% of drivers active during peak hours
    7. Location updates every 5 seconds during active delivery
    8. Eventual consistency acceptable for non-critical updates

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Geospatial indexing:** Fast proximity searches using geohashing
    2. **Real-time matching:** Sub-30 second driver assignment using dispatch algorithms
    3. **Event-driven architecture:** Async processing for status updates, notifications
    4. **Write-heavy optimization:** 410K write QPS (location updates dominate)
    5. **Eventual consistency:** Status updates may lag by 1-2 seconds (acceptable)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Customer[Customer App]
            Driver[Driver App]
            Restaurant[Restaurant Dashboard]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Restaurant images/menus]
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Search_API[Search Service<br/>Restaurant discovery]
            Order_API[Order Service<br/>Place/manage orders]
            Matching_API[Matching Service<br/>Driver dispatch]
            Tracking_API[Tracking Service<br/>Real-time location]
            Payment_API[Payment Service<br/>Transactions]
            ETA_API[ETA Service<br/>Route & time prediction]
            Pricing_API[Pricing Service<br/>Surge pricing]
        end

        subgraph "Core Services"
            Dispatch[Dispatch Engine<br/>Matching algorithm]
            Routing[Routing Engine<br/>GraphHopper/Valhalla]
            GeoIndex[Geo Index Service<br/>Geohashing]
            Notification[Notification Service<br/>Push/SMS/Email]
        end

        subgraph "Data Processing"
            Location_Stream[Location Processor<br/>Driver positions]
            Order_Stream[Order Processor<br/>Status updates]
            Analytics[Analytics Pipeline<br/>Demand forecasting]
            ML_ETA[ML ETA Model<br/>Prediction service]
        end

        subgraph "Caching"
            Redis_Geo[Redis Geospatial<br/>Driver locations]
            Redis_Rest[Redis<br/>Restaurant cache]
            Redis_Order[Redis<br/>Active orders]
            Redis_Route[Redis<br/>Route cache]
        end

        subgraph "Storage"
            Order_DB[(Order DB<br/>PostgreSQL<br/>Sharded)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Restaurant_DB[(Restaurant DB<br/>PostgreSQL)]
            Driver_DB[(Driver DB<br/>PostgreSQL)]
            Location_DB[(Location DB<br/>TimescaleDB<br/>Time-series)]
            Search_DB[(Elasticsearch<br/>Restaurant search)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "External"
            Maps_API[Google Maps API<br/>Geocoding/routing]
            Payment_GW[Payment Gateway<br/>Stripe/PayPal]
        end

        Customer --> CDN
        Driver --> CDN
        Restaurant --> CDN
        Customer --> LB
        Driver --> LB
        Restaurant --> LB

        CDN --> S3[S3<br/>Restaurant media]

        LB --> Search_API
        LB --> Order_API
        LB --> Matching_API
        LB --> Tracking_API
        LB --> Payment_API
        LB --> ETA_API
        LB --> Pricing_API

        Search_API --> Redis_Rest
        Search_API --> Search_DB
        Search_API --> GeoIndex

        Order_API --> Kafka
        Order_API --> Order_DB
        Order_API --> Redis_Order

        Matching_API --> Dispatch
        Dispatch --> Redis_Geo
        Dispatch --> GeoIndex

        Tracking_API --> Redis_Geo
        Tracking_API --> Location_DB

        Driver --> Tracking_API
        Tracking_API --> Location_Stream

        Payment_API --> Payment_GW
        Payment_API --> Order_DB

        ETA_API --> Routing
        ETA_API --> Redis_Route
        ETA_API --> ML_ETA

        Pricing_API --> Analytics
        Pricing_API --> Redis_Order

        Kafka --> Location_Stream
        Kafka --> Order_Stream
        Kafka --> Notification
        Kafka --> Analytics

        Location_Stream --> Redis_Geo
        Location_Stream --> Location_DB

        Order_Stream --> Redis_Order
        Order_Stream --> Order_DB

        Routing --> Maps_API

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Geo fill:#fff4e1
        style Redis_Rest fill:#fff4e1
        style Redis_Order fill:#fff4e1
        style Redis_Route fill:#fff4e1
        style Order_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Restaurant_DB fill:#ffe1e1
        style Location_DB fill:#e1f5e1
        style Search_DB fill:#e8eaf6
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis Geospatial** | Fast proximity queries (< 10ms), GEORADIUS for nearby drivers | PostGIS (too slow for 400K updates/sec), custom geohashing (complex) |
    | **PostgreSQL (Order DB)** | ACID transactions for orders, strong consistency for payments | MongoDB (weak consistency), Cassandra (no transactions) |
    | **TimescaleDB** | Optimized for time-series location data, fast range queries | Cassandra (slower queries), InfluxDB (limited SQL) |
    | **Kafka** | High-throughput event streaming (410K write QPS), reliable delivery | RabbitMQ (can't handle throughput), direct updates (no replay) |
    | **GraphHopper** | Fast routing (<50ms), offline maps, customizable | Google Directions API (expensive at scale), custom routing (complex) |
    | **Elasticsearch** | Fast restaurant search, autocomplete, filters | Database LIKE queries (too slow), custom index (reinventing wheel) |

    **Key Trade-off:** We chose **availability over consistency** for location updates. Driver positions may lag by 1-2 seconds, but system remains available during failures.

    ---

    ## API Design

    ### 1. Search Restaurants

    **Request:**
    ```http
    GET /api/v1/restaurants/search?lat=37.7749&lon=-122.4194&radius=5&cuisine=italian&sort=rating
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "restaurants": [
        {
          "restaurant_id": "rest_123",
          "name": "Luigi's Pizza",
          "cuisine": ["italian", "pizza"],
          "rating": 4.7,
          "review_count": 1542,
          "distance_miles": 0.8,
          "delivery_time_min": 30,
          "delivery_fee": 2.99,
          "minimum_order": 15.00,
          "is_open": true,
          "image_url": "https://cdn.ubereats.com/rest_123.jpg",
          "location": {
            "lat": 37.7755,
            "lon": -122.4183,
            "address": "123 Market St, San Francisco, CA 94103"
          }
        }
        // ... more restaurants
      ],
      "total": 47,
      "cursor": "eyJvZmZzZXQiOjIwfQ=="
    }
    ```

    **Design Notes:**

    - Geospatial search using geohashing + Elasticsearch
    - Cache popular searches (lat/lon rounded to 0.01 degrees)
    - Filter by: cuisine, rating, delivery time, price range
    - Sort by: distance, rating, delivery time, popularity

    ---

    ### 2. Place Order

    **Request:**
    ```http
    POST /api/v1/orders
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "restaurant_id": "rest_123",
      "items": [
        {
          "item_id": "item_456",
          "name": "Margherita Pizza",
          "quantity": 1,
          "price": 14.99,
          "customizations": ["extra cheese", "no onions"]
        },
        {
          "item_id": "item_789",
          "name": "Caesar Salad",
          "quantity": 1,
          "price": 8.99
        }
      ],
      "delivery_address": {
        "lat": 37.7749,
        "lon": -122.4194,
        "street": "456 Mission St",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94105",
        "instructions": "Ring doorbell"
      },
      "payment_method_id": "pm_123",
      "delivery_option": "standard",  // or "scheduled"
      "scheduled_time": null,
      "promo_code": "SUMMER20"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "order_id": "order_789",
      "status": "pending_restaurant_confirmation",
      "restaurant": {
        "restaurant_id": "rest_123",
        "name": "Luigi's Pizza"
      },
      "items": [...],
      "pricing": {
        "subtotal": 23.98,
        "delivery_fee": 2.99,
        "service_fee": 1.50,
        "tax": 2.15,
        "discount": -4.80,
        "total": 25.82
      },
      "estimated_delivery_time": "2026-02-02T19:45:00Z",
      "created_at": "2026-02-02T19:00:00Z"
    }
    ```

    **Design Notes:**

    - Create order in database with PENDING status
    - Validate restaurant is open and accepting orders
    - Calculate pricing with surge/promo codes
    - Publish order_created event to Kafka
    - Return immediately (don't wait for driver matching)
    - Idempotency key to prevent duplicate orders

    ---

    ### 3. Match Driver (Internal API)

    **Request:**
    ```http
    POST /api/internal/v1/dispatch/match
    Content-Type: application/json

    {
      "order_id": "order_789",
      "restaurant_location": {
        "lat": 37.7755,
        "lon": -122.4183
      },
      "delivery_location": {
        "lat": 37.7749,
        "lon": -122.4194
      },
      "ready_time": "2026-02-02T19:20:00Z"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "driver_id": "driver_456",
      "driver_name": "John Smith",
      "driver_location": {
        "lat": 37.7760,
        "lon": -122.4175
      },
      "rating": 4.9,
      "vehicle": "Toyota Prius",
      "eta_to_restaurant": 8,  // minutes
      "estimated_pickup_time": "2026-02-02T19:28:00Z",
      "estimated_delivery_time": "2026-02-02T19:45:00Z",
      "match_score": 0.92
    }
    ```

    **Design Notes:**

    - Use dispatch algorithm (see Deep Dive section)
    - Find drivers within 2-mile radius of restaurant
    - Consider: distance, driver rating, acceptance rate
    - Fallback: expand radius or wait for driver availability
    - Timeout: 30 seconds (escalate if no match)

    ---

    ### 4. Update Location (Driver App)

    **Request:**
    ```http
    POST /api/v1/drivers/location
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "driver_id": "driver_456",
      "lat": 37.7765,
      "lon": -122.4180,
      "accuracy": 10,  // meters
      "heading": 45,   // degrees
      "speed": 25,     // mph
      "timestamp": "2026-02-02T19:32:15Z"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 204 No Content
    ```

    **Design Notes:**

    - High-frequency updates (every 5 seconds during delivery)
    - Update Redis geospatial index for real-time matching
    - Publish location_updated event for tracking
    - Batch write to TimescaleDB for history
    - No response body (minimize bandwidth)

    ---

    ### 5. Track Order (Customer App)

    **Request:**
    ```http
    GET /api/v1/orders/order_789/track
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "order_id": "order_789",
      "status": "in_transit",
      "driver": {
        "driver_id": "driver_456",
        "name": "John Smith",
        "rating": 4.9,
        "vehicle": "Toyota Prius",
        "photo_url": "https://cdn.ubereats.com/drivers/456.jpg",
        "location": {
          "lat": 37.7765,
          "lon": -122.4180,
          "heading": 45,
          "last_updated": "2026-02-02T19:32:15Z"
        }
      },
      "timeline": [
        {
          "status": "placed",
          "timestamp": "2026-02-02T19:00:00Z"
        },
        {
          "status": "confirmed",
          "timestamp": "2026-02-02T19:02:00Z"
        },
        {
          "status": "preparing",
          "timestamp": "2026-02-02T19:05:00Z"
        },
        {
          "status": "ready_for_pickup",
          "timestamp": "2026-02-02T19:25:00Z"
        },
        {
          "status": "picked_up",
          "timestamp": "2026-02-02T19:28:00Z"
        },
        {
          "status": "in_transit",
          "timestamp": "2026-02-02T19:30:00Z"
        }
      ],
      "eta": "2026-02-02T19:45:00Z",
      "eta_confidence": 0.85
    }
    ```

    **Design Notes:**

    - Real-time driver location from Redis
    - ETA updated every 30 seconds
    - WebSocket for live updates (optional)
    - Polyline route visualization

    ---

    ## Database Schema

    ### Orders (PostgreSQL - Sharded by order_id)

    ```sql
    -- Orders table
    CREATE TABLE orders (
        order_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        restaurant_id VARCHAR(50) NOT NULL,
        driver_id VARCHAR(50),
        status VARCHAR(50) NOT NULL,  -- pending, confirmed, preparing, ready, picked_up, in_transit, delivered, cancelled

        -- Locations
        restaurant_lat DECIMAL(10, 8) NOT NULL,
        restaurant_lon DECIMAL(11, 8) NOT NULL,
        delivery_lat DECIMAL(10, 8) NOT NULL,
        delivery_lon DECIMAL(11, 8) NOT NULL,
        delivery_address JSONB NOT NULL,

        -- Pricing
        subtotal DECIMAL(10, 2) NOT NULL,
        delivery_fee DECIMAL(10, 2) NOT NULL,
        service_fee DECIMAL(10, 2) NOT NULL,
        tax DECIMAL(10, 2) NOT NULL,
        discount DECIMAL(10, 2) DEFAULT 0,
        total DECIMAL(10, 2) NOT NULL,

        -- Items
        items JSONB NOT NULL,  -- Array of {item_id, name, quantity, price, customizations}

        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confirmed_at TIMESTAMP,
        ready_at TIMESTAMP,
        picked_up_at TIMESTAMP,
        delivered_at TIMESTAMP,

        -- Estimates
        estimated_prep_time INT,  -- minutes
        estimated_delivery_time TIMESTAMP,

        INDEX idx_user_id (user_id),
        INDEX idx_restaurant_id (restaurant_id),
        INDEX idx_driver_id (driver_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    ) PARTITION BY HASH (order_id);

    -- Order items (denormalized in orders.items JSONB for simplicity)
    -- But can be separate table for complex queries
    ```

    ---

    ### Restaurants (PostgreSQL)

    ```sql
    CREATE TABLE restaurants (
        restaurant_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        cuisine VARCHAR(100)[] NOT NULL,  -- Array: ['italian', 'pizza']

        -- Location
        lat DECIMAL(10, 8) NOT NULL,
        lon DECIMAL(11, 8) NOT NULL,
        address JSONB NOT NULL,
        geohash VARCHAR(12),  -- For geospatial indexing

        -- Operational
        is_open BOOLEAN DEFAULT true,
        operating_hours JSONB,  -- {monday: {open: '09:00', close: '22:00'}, ...}
        average_prep_time INT,  -- minutes

        -- Pricing & Ratings
        price_range INT,  -- 1-4 ($, $$, $$$, $$$$)
        minimum_order DECIMAL(10, 2),
        delivery_fee DECIMAL(10, 2),
        rating DECIMAL(3, 2),
        review_count INT DEFAULT 0,

        -- Media
        image_url TEXT,
        logo_url TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_geohash (geohash),
        INDEX idx_cuisine (cuisine),
        INDEX idx_rating (rating)
    );

    -- Menu items
    CREATE TABLE menu_items (
        item_id VARCHAR(50) PRIMARY KEY,
        restaurant_id VARCHAR(50) REFERENCES restaurants(restaurant_id),
        name VARCHAR(255) NOT NULL,
        description TEXT,
        category VARCHAR(100),  -- appetizer, entree, dessert, drink
        price DECIMAL(10, 2) NOT NULL,
        image_url TEXT,
        is_available BOOLEAN DEFAULT true,
        customization_options JSONB,  -- [{name: 'Size', options: ['Small', 'Large']}, ...]

        INDEX idx_restaurant_id (restaurant_id),
        INDEX idx_category (category)
    );
    ```

    ---

    ### Drivers (PostgreSQL)

    ```sql
    CREATE TABLE drivers (
        driver_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        phone VARCHAR(20) NOT NULL,

        -- Vehicle
        vehicle_type VARCHAR(50),  -- car, bike, scooter
        vehicle_model VARCHAR(100),
        license_plate VARCHAR(20),

        -- Status
        is_active BOOLEAN DEFAULT true,
        current_status VARCHAR(50),  -- available, busy, offline
        current_lat DECIMAL(10, 8),
        current_lon DECIMAL(11, 8),
        last_location_update TIMESTAMP,

        -- Ratings
        rating DECIMAL(3, 2),
        total_deliveries INT DEFAULT 0,
        acceptance_rate DECIMAL(5, 2),  -- Percentage
        completion_rate DECIMAL(5, 2),

        -- Media
        photo_url TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_status (current_status),
        INDEX idx_rating (rating)
    );
    ```

    ---

    ### Location History (TimescaleDB)

    ```sql
    -- TimescaleDB extension for time-series data
    CREATE EXTENSION IF NOT EXISTS timescaledb;

    CREATE TABLE driver_locations (
        driver_id VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        lat DECIMAL(10, 8) NOT NULL,
        lon DECIMAL(11, 8) NOT NULL,
        accuracy INT,  -- meters
        heading INT,   -- degrees (0-360)
        speed INT,     -- mph
        order_id VARCHAR(50),  -- if on active delivery

        PRIMARY KEY (driver_id, timestamp)
    );

    -- Convert to hypertable (TimescaleDB)
    SELECT create_hypertable('driver_locations', 'timestamp');

    -- Create index for queries
    CREATE INDEX idx_driver_id_time ON driver_locations (driver_id, timestamp DESC);
    ```

    ---

    ## Data Flow Diagrams

    ### Order Placement & Driver Matching Flow

    ```mermaid
    sequenceDiagram
        participant Customer
        participant Order_API
        participant Order_DB
        participant Kafka
        participant Matching_Service
        participant Dispatch_Engine
        participant Redis_Geo
        participant Driver_App
        participant Notification

        Customer->>Order_API: POST /api/v1/orders
        Order_API->>Order_API: Validate restaurant, calculate pricing
        Order_API->>Order_DB: INSERT order (status: pending)
        Order_DB-->>Order_API: order_id

        Order_API->>Kafka: Publish order_created event
        Order_API-->>Customer: 201 Created (order_id, ETA)

        Kafka->>Matching_Service: Process order_created

        alt Restaurant auto-confirms
            Matching_Service->>Order_DB: UPDATE status = confirmed
            Matching_Service->>Kafka: Publish order_confirmed
        else Restaurant manual confirmation
            Matching_Service->>Notification: Notify restaurant
            Note over Notification: Wait for restaurant confirmation
        end

        Kafka->>Dispatch_Engine: Process order_confirmed
        Dispatch_Engine->>Redis_Geo: GEORADIUS (find nearby drivers)
        Redis_Geo-->>Dispatch_Engine: Available drivers list

        Dispatch_Engine->>Dispatch_Engine: Run matching algorithm<br/>(score drivers)
        Dispatch_Engine->>Dispatch_Engine: Select best driver

        Dispatch_Engine->>Driver_App: Send order offer
        Driver_App-->>Dispatch_Engine: Accept offer

        alt Driver accepts
            Dispatch_Engine->>Order_DB: UPDATE driver_id, status = preparing
            Dispatch_Engine->>Notification: Notify customer (driver assigned)
        else Driver rejects or timeout (20s)
            Dispatch_Engine->>Dispatch_Engine: Try next driver
            Note over Dispatch_Engine: Escalate after 3 attempts
        end
    ```

    ---

    ### Real-time Tracking Flow

    ```mermaid
    sequenceDiagram
        participant Driver_App
        participant Tracking_API
        participant Kafka
        participant Location_Stream
        participant Redis_Geo
        participant Location_DB
        participant Customer_App
        participant WebSocket

        loop Every 5 seconds
            Driver_App->>Tracking_API: POST /drivers/location
            Tracking_API->>Kafka: Publish location_updated event
            Tracking_API-->>Driver_App: 204 No Content

            Kafka->>Location_Stream: Process location_updated

            par Update real-time cache
                Location_Stream->>Redis_Geo: GEOADD (update driver position)
            and Persist to database
                Location_Stream->>Location_DB: INSERT location (batch)
            and Notify customer
                Location_Stream->>WebSocket: Push location to customer
                WebSocket->>Customer_App: Real-time location update
            end
        end

        Customer_App->>Tracking_API: GET /orders/{id}/track
        Tracking_API->>Redis_Geo: GEOPOS (get driver location)
        Redis_Geo-->>Tracking_API: Current location
        Tracking_API-->>Customer_App: Driver location + ETA
    ```

    ---

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical food delivery subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Driver Matching** | How to match driver to order in <30s? | Dispatch algorithm with scoring (distance, rating, acceptance rate) |
    | **Route Optimization** | How to calculate fastest route with traffic? | GraphHopper + historical traffic data + ML predictions |
    | **ETA Prediction** | How to predict delivery time accurately? | ML model trained on historical data (prep time, traffic, distance) |
    | **Surge Pricing** | How to balance supply/demand? | Dynamic pricing based on driver availability, order density |

    ---

    === "🚗 Driver Matching & Dispatch"

        ## The Challenge

        **Problem:** Match driver to order within 30 seconds, considering distance, availability, rating, and future demand.

        **Complexity:**

        - 2M active drivers during peak hours
        - 580 orders/sec (2,900 during peak)
        - Multiple constraints: proximity, capacity, driver preferences
        - Optimize for: delivery time, driver utilization, customer satisfaction

        **Naive approach:** Assign nearest available driver. **Problem:** Leads to imbalanced distribution, longer average delivery times.

        ---

        ## Dispatch Algorithm (Scoring-Based)

        **Multi-factor scoring approach:**

        ```
        MatchScore(driver, order) =
            α × DistanceScore +
            β × ETAScore +
            γ × DriverQualityScore +
            δ × AcceptanceScore +
            ε × FutureOpportunityScore

        where α + β + γ + δ + ε = 1 (weights sum to 1)
        ```

        **Typical weights:**
        - α (Distance): 0.35
        - β (ETA): 0.25
        - γ (Driver Quality): 0.20
        - δ (Acceptance Rate): 0.10
        - ε (Future Opportunity): 0.10

        ---

        ## Implementation

        ```python
        import math
        from typing import List, Dict, Optional
        from dataclasses import dataclass
        from redis import Redis
        from datetime import datetime, timedelta

        @dataclass
        class Order:
            order_id: str
            restaurant_lat: float
            restaurant_lon: float
            delivery_lat: float
            delivery_lon: float
            ready_time: datetime
            estimated_prep_time: int  # minutes
            priority: int  # 1-5, higher = more urgent

        @dataclass
        class Driver:
            driver_id: str
            lat: float
            lon: float
            rating: float  # 1.0-5.0
            acceptance_rate: float  # 0.0-1.0
            completion_rate: float
            total_deliveries: int
            current_status: str  # available, busy, offline

        class DispatchEngine:
            """Driver matching with multi-factor scoring"""

            # Weights for scoring algorithm
            WEIGHT_DISTANCE = 0.35
            WEIGHT_ETA = 0.25
            WEIGHT_QUALITY = 0.20
            WEIGHT_ACCEPTANCE = 0.10
            WEIGHT_OPPORTUNITY = 0.10

            # Constraints
            MAX_SEARCH_RADIUS_MILES = 5.0
            MIN_ACCEPTANCE_RATE = 0.70
            OFFER_TIMEOUT_SECONDS = 20

            def __init__(self, redis_client: Redis, routing_service, analytics_service):
                self.redis = redis_client
                self.routing = routing_service
                self.analytics = analytics_service

            def find_best_driver(self, order: Order) -> Optional[Driver]:
                """
                Find best driver for order using scoring algorithm

                Args:
                    order: Order object with pickup/delivery locations

                Returns:
                    Best matched driver or None if no match found
                """
                # Step 1: Get nearby available drivers
                nearby_drivers = self._get_nearby_drivers(
                    order.restaurant_lat,
                    order.restaurant_lon,
                    radius_miles=self.MAX_SEARCH_RADIUS_MILES
                )

                if not nearby_drivers:
                    # Expand search radius or wait
                    nearby_drivers = self._get_nearby_drivers(
                        order.restaurant_lat,
                        order.restaurant_lon,
                        radius_miles=self.MAX_SEARCH_RADIUS_MILES * 2
                    )

                if not nearby_drivers:
                    return None

                # Step 2: Filter by constraints
                eligible_drivers = [
                    driver for driver in nearby_drivers
                    if driver.current_status == 'available'
                    and driver.acceptance_rate >= self.MIN_ACCEPTANCE_RATE
                ]

                if not eligible_drivers:
                    return None

                # Step 3: Score each driver
                scored_drivers = []
                for driver in eligible_drivers:
                    score = self._calculate_match_score(driver, order)
                    scored_drivers.append((driver, score))

                # Step 4: Sort by score (descending)
                scored_drivers.sort(key=lambda x: x[1], reverse=True)

                # Step 5: Try drivers in order until one accepts
                for driver, score in scored_drivers[:5]:  # Try top 5
                    accepted = self._offer_order_to_driver(driver, order, score)
                    if accepted:
                        return driver

                return None

            def _get_nearby_drivers(self, lat: float, lon: float, radius_miles: float) -> List[Driver]:
                """
                Get available drivers within radius using Redis Geospatial

                Args:
                    lat, lon: Center point
                    radius_miles: Search radius

                Returns:
                    List of nearby drivers
                """
                # Convert miles to meters
                radius_meters = radius_miles * 1609.34

                # Redis GEORADIUS command
                # Returns: [(driver_id, distance), ...]
                results = self.redis.georadius(
                    name='driver_locations',
                    longitude=lon,
                    latitude=lat,
                    radius=radius_meters,
                    unit='m',
                    withdist=True,
                    sort='ASC'  # Nearest first
                )

                # Fetch driver details
                drivers = []
                for driver_id, distance in results:
                    driver_data = self.redis.hgetall(f'driver:{driver_id}')
                    if driver_data:
                        drivers.append(Driver(
                            driver_id=driver_id,
                            lat=float(driver_data['lat']),
                            lon=float(driver_data['lon']),
                            rating=float(driver_data['rating']),
                            acceptance_rate=float(driver_data['acceptance_rate']),
                            completion_rate=float(driver_data['completion_rate']),
                            total_deliveries=int(driver_data['total_deliveries']),
                            current_status=driver_data['status']
                        ))

                return drivers

            def _calculate_match_score(self, driver: Driver, order: Order) -> float:
                """
                Calculate match score for driver-order pair

                Returns:
                    Score between 0.0 and 1.0 (higher = better match)
                """
                # 1. Distance score (closer = better)
                distance_score = self._calculate_distance_score(driver, order)

                # 2. ETA score (faster = better)
                eta_score = self._calculate_eta_score(driver, order)

                # 3. Driver quality score (higher rating = better)
                quality_score = self._calculate_quality_score(driver)

                # 4. Acceptance rate score
                acceptance_score = driver.acceptance_rate

                # 5. Future opportunity score (avoid stranding driver in low-demand area)
                opportunity_score = self._calculate_opportunity_score(driver, order)

                # Weighted sum
                total_score = (
                    self.WEIGHT_DISTANCE * distance_score +
                    self.WEIGHT_ETA * eta_score +
                    self.WEIGHT_QUALITY * quality_score +
                    self.WEIGHT_ACCEPTANCE * acceptance_score +
                    self.WEIGHT_OPPORTUNITY * opportunity_score
                )

                return total_score

            def _calculate_distance_score(self, driver: Driver, order: Order) -> float:
                """
                Score based on driver distance to restaurant

                Returns:
                    Score 0.0-1.0 (1.0 = closest)
                """
                distance_miles = self._haversine_distance(
                    driver.lat, driver.lon,
                    order.restaurant_lat, order.restaurant_lon
                )

                # Exponential decay: score drops as distance increases
                # 0 miles = 1.0, 5 miles = ~0.05
                score = math.exp(-distance_miles / 2.0)
                return score

            def _calculate_eta_score(self, driver: Driver, order: Order) -> float:
                """
                Score based on estimated pickup time

                Returns:
                    Score 0.0-1.0 (1.0 = fastest)
                """
                # Get route from driver -> restaurant
                eta_minutes = self.routing.calculate_eta(
                    from_lat=driver.lat,
                    from_lon=driver.lon,
                    to_lat=order.restaurant_lat,
                    to_lon=order.restaurant_lon
                )

                # Target: arrive when food is ready
                food_ready_in = order.estimated_prep_time  # minutes
                time_difference = abs(eta_minutes - food_ready_in)

                # Perfect timing = 1.0, large difference = lower score
                score = math.exp(-time_difference / 10.0)
                return score

            def _calculate_quality_score(self, driver: Driver) -> float:
                """
                Score based on driver rating and experience

                Returns:
                    Score 0.0-1.0
                """
                # Normalize rating (5.0 -> 1.0, 3.0 -> 0.0)
                rating_normalized = (driver.rating - 3.0) / 2.0

                # Experience bonus (more deliveries = higher score)
                experience_factor = min(driver.total_deliveries / 1000.0, 1.0)

                # Combine
                score = (0.7 * rating_normalized) + (0.3 * experience_factor)
                return max(0.0, min(score, 1.0))

            def _calculate_opportunity_score(self, driver: Driver, order: Order) -> float:
                """
                Score based on future order availability near delivery location

                Prevents stranding drivers in low-demand areas

                Returns:
                    Score 0.0-1.0 (1.0 = high demand area)
                """
                # Get predicted demand at delivery location
                demand_score = self.analytics.get_demand_prediction(
                    lat=order.delivery_lat,
                    lon=order.delivery_lon,
                    time_ahead_minutes=30
                )

                return demand_score

            def _offer_order_to_driver(self, driver: Driver, order: Order, score: float) -> bool:
                """
                Send order offer to driver and wait for acceptance

                Args:
                    driver: Driver to offer order
                    order: Order details
                    score: Match score (for logging)

                Returns:
                    True if driver accepted, False otherwise
                """
                # Create offer
                offer = {
                    'order_id': order.order_id,
                    'restaurant_lat': order.restaurant_lat,
                    'restaurant_lon': order.restaurant_lon,
                    'delivery_lat': order.delivery_lat,
                    'delivery_lon': order.delivery_lon,
                    'estimated_earnings': self._calculate_earnings(order),
                    'expires_at': datetime.utcnow() + timedelta(seconds=self.OFFER_TIMEOUT_SECONDS)
                }

                # Send push notification to driver app
                self._send_driver_notification(driver.driver_id, offer)

                # Wait for response (using Redis pub/sub or polling)
                response = self._wait_for_driver_response(
                    driver.driver_id,
                    order.order_id,
                    timeout_seconds=self.OFFER_TIMEOUT_SECONDS
                )

                if response == 'accepted':
                    logger.info(f"Driver {driver.driver_id} accepted order {order.order_id} (score: {score:.2f})")
                    return True
                else:
                    logger.info(f"Driver {driver.driver_id} rejected/timeout order {order.order_id}")
                    return False

            @staticmethod
            def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
                """
                Calculate distance between two points using Haversine formula

                Returns:
                    Distance in miles
                """
                R = 3959.0  # Earth radius in miles

                lat1_rad = math.radians(lat1)
                lat2_rad = math.radians(lat2)
                delta_lat = math.radians(lat2 - lat1)
                delta_lon = math.radians(lon2 - lon1)

                a = (math.sin(delta_lat / 2) ** 2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) *
                     math.sin(delta_lon / 2) ** 2)
                c = 2 * math.asin(math.sqrt(a))

                return R * c
        ```

        ---

        ## Geohashing for Spatial Indexing

        **Problem:** Efficiently find nearby drivers from 2M active drivers.

        **Solution:** Geohashing converts (lat, lon) to short string for quick prefix matching.

        ```python
        import geohash2

        class GeoIndexService:
            """Geohashing for fast proximity queries"""

            # Geohash precision levels
            # precision=6: ~1.2km x 0.6km cell
            # precision=7: ~153m x 153m cell
            GEOHASH_PRECISION = 6

            def __init__(self, redis_client):
                self.redis = redis_client

            def update_driver_location(self, driver_id: str, lat: float, lon: float):
                """
                Update driver location in geohash index

                Args:
                    driver_id: Driver ID
                    lat, lon: Location coordinates
                """
                # Calculate geohash
                gh = geohash2.encode(lat, lon, precision=self.GEOHASH_PRECISION)

                # Add to geohash set in Redis
                self.redis.sadd(f'geohash:{gh}', driver_id)

                # Store driver's current geohash
                old_geohash = self.redis.hget(f'driver:{driver_id}', 'geohash')
                if old_geohash and old_geohash != gh:
                    # Remove from old geohash
                    self.redis.srem(f'geohash:{old_geohash}', driver_id)

                # Update driver's geohash
                self.redis.hset(f'driver:{driver_id}', 'geohash', gh)

            def get_nearby_drivers(self, lat: float, lon: float, radius_miles: float) -> List[str]:
                """
                Get drivers within radius using geohash

                Args:
                    lat, lon: Center point
                    radius_miles: Search radius

                Returns:
                    List of driver IDs
                """
                # Get geohash neighbors
                center_gh = geohash2.encode(lat, lon, precision=self.GEOHASH_PRECISION)
                neighbors = geohash2.neighbors(center_gh)
                geohashes = [center_gh] + neighbors

                # Get all drivers in these geohashes
                driver_ids = set()
                for gh in geohashes:
                    drivers = self.redis.smembers(f'geohash:{gh}')
                    driver_ids.update(drivers)

                # Filter by actual distance (geohash is approximation)
                nearby_drivers = []
                for driver_id in driver_ids:
                    driver_data = self.redis.hgetall(f'driver:{driver_id}')
                    if driver_data:
                        driver_lat = float(driver_data['lat'])
                        driver_lon = float(driver_data['lon'])
                        distance = self._haversine_distance(lat, lon, driver_lat, driver_lon)
                        if distance <= radius_miles:
                            nearby_drivers.append(driver_id)

                return nearby_drivers
        ```

        ---

        ## Batching & Optimization

        **Problem:** During peak (2,900 orders/sec), matching each order individually is inefficient.

        **Solution:** Batch matching every 5 seconds.

        ```python
        class BatchDispatcher:
            """Batch driver matching for peak efficiency"""

            BATCH_INTERVAL_SECONDS = 5

            def __init__(self, dispatch_engine):
                self.dispatch = dispatch_engine
                self.pending_orders = []

            def add_order(self, order: Order):
                """Add order to pending batch"""
                self.pending_orders.append(order)

            def run_batch_matching(self):
                """
                Match all pending orders in batch

                Optimization: Solve as assignment problem (Hungarian algorithm)
                """
                if not self.pending_orders:
                    return

                # Get all available drivers
                all_drivers = self._get_all_available_drivers()

                # Build cost matrix
                cost_matrix = []
                for order in self.pending_orders:
                    order_costs = []
                    for driver in all_drivers:
                        score = self.dispatch._calculate_match_score(driver, order)
                        cost = 1.0 - score  # Convert score to cost (lower = better)
                        order_costs.append(cost)
                    cost_matrix.append(order_costs)

                # Solve assignment problem
                from scipy.optimize import linear_sum_assignment
                order_indices, driver_indices = linear_sum_assignment(cost_matrix)

                # Assign drivers to orders
                for order_idx, driver_idx in zip(order_indices, driver_indices):
                    order = self.pending_orders[order_idx]
                    driver = all_drivers[driver_idx]
                    self._assign_driver_to_order(driver, order)

                # Clear pending orders
                self.pending_orders = []
        ```

    === "🗺️ Route Optimization & Navigation"

        ## The Challenge

        **Problem:** Calculate fastest route from driver -> restaurant -> customer, considering:

        - Real-time traffic conditions
        - Road restrictions (one-way, no left turn, etc.)
        - Time of day (rush hour vs off-peak)
        - Historical patterns

        **Requirements:**

        - **Fast:** < 50ms for route calculation
        - **Accurate:** Within 10% of actual travel time
        - **Real-time:** Update route based on traffic
        - **Cost-effective:** Google Directions API costs $5 per 1000 requests

        ---

        ## Routing Architecture

        **Two-tier approach:**

        1. **Self-hosted routing:** GraphHopper or Valhalla (open-source)
        2. **External API fallback:** Google Maps API for complex cases

        **Why self-hosted:**

        - **Cost:** $0 vs $5 per 1000 requests
        - **Scale:** 50M orders/day = 100M route calculations = $500K/day with Google
        - **Control:** Customize routing logic, priority lanes
        - **Latency:** < 50ms (local) vs 200ms (API)

        ---

        ## GraphHopper Implementation

        ```python
        import requests
        from typing import List, Tuple, Optional
        from dataclasses import dataclass
        from datetime import datetime

        @dataclass
        class RouteSegment:
            from_lat: float
            from_lon: float
            to_lat: float
            to_lon: float
            distance_meters: float
            duration_seconds: int
            instructions: List[str]
            polyline: str  # Encoded polyline for map visualization

        class RoutingEngine:
            """Route calculation using self-hosted GraphHopper"""

            def __init__(self, graphhopper_url: str, cache_client):
                self.graphhopper_url = graphhopper_url
                self.cache = cache_client

            def calculate_route(
                self,
                waypoints: List[Tuple[float, float]],
                vehicle: str = 'car',
                optimize: bool = True
            ) -> Optional[List[RouteSegment]]:
                """
                Calculate optimal route through multiple waypoints

                Args:
                    waypoints: List of (lat, lon) tuples
                    vehicle: car, bike, scooter
                    optimize: Whether to optimize waypoint order

                Returns:
                    List of route segments
                """
                # Check cache
                cache_key = self._get_cache_key(waypoints, vehicle)
                cached = self.cache.get(cache_key)
                if cached:
                    return cached

                # Build GraphHopper request
                params = {
                    'point': [f"{lat},{lon}" for lat, lon in waypoints],
                    'vehicle': vehicle,
                    'optimize': 'true' if optimize else 'false',
                    'points_encoded': 'true',  # Polyline encoding
                    'instructions': 'true',
                    'calc_points': 'true'
                }

                # Call GraphHopper API
                response = requests.get(
                    f"{self.graphhopper_url}/route",
                    params=params,
                    timeout=5
                )

                if response.status_code != 200:
                    logger.error(f"GraphHopper error: {response.text}")
                    return None

                data = response.json()
                if 'paths' not in data or not data['paths']:
                    return None

                # Parse response
                path = data['paths'][0]
                segments = self._parse_route_segments(path)

                # Cache for 5 minutes (routes change with traffic)
                self.cache.setex(cache_key, 300, segments)

                return segments

            def calculate_eta(
                self,
                from_lat: float,
                from_lon: float,
                to_lat: float,
                to_lon: float,
                departure_time: Optional[datetime] = None
            ) -> int:
                """
                Calculate ETA in minutes

                Args:
                    from_lat, from_lon: Starting point
                    to_lat, to_lon: Destination
                    departure_time: Estimated departure (for traffic prediction)

                Returns:
                    ETA in minutes
                """
                route = self.calculate_route([(from_lat, from_lon), (to_lat, to_lon)])
                if not route:
                    # Fallback: straight-line distance / average speed
                    distance_miles = self._haversine_distance(from_lat, from_lon, to_lat, to_lon)
                    return int(distance_miles / 0.5)  # 30 mph average

                total_duration = sum(seg.duration_seconds for seg in route)

                # Adjust for traffic (if departure time provided)
                if departure_time:
                    traffic_multiplier = self._get_traffic_multiplier(
                        from_lat, from_lon,
                        departure_time
                    )
                    total_duration *= traffic_multiplier

                return int(total_duration / 60)  # Convert to minutes

            def _get_traffic_multiplier(self, lat: float, lon: float, time: datetime) -> float:
                """
                Get traffic multiplier based on historical data

                Returns:
                    Multiplier (1.0 = no traffic, 2.0 = double time)
                """
                hour = time.hour
                day_of_week = time.weekday()

                # Rush hour multipliers (simplified)
                if day_of_week < 5:  # Weekday
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        return 1.5  # 50% longer during rush hour
                    elif 12 <= hour <= 13:
                        return 1.2  # 20% longer during lunch
                    else:
                        return 1.0
                else:  # Weekend
                    if 11 <= hour <= 14:
                        return 1.3  # Brunch/lunch traffic
                    else:
                        return 0.9  # Generally faster on weekends

            def _parse_route_segments(self, path: dict) -> List[RouteSegment]:
                """Parse GraphHopper response into route segments"""
                segments = []
                instructions = path.get('instructions', [])

                for i, instruction in enumerate(instructions):
                    interval = instruction['interval']
                    points = path['points']['coordinates'][interval[0]:interval[1]+1]

                    if len(points) >= 2:
                        segment = RouteSegment(
                            from_lat=points[0][1],
                            from_lon=points[0][0],
                            to_lat=points[-1][1],
                            to_lon=points[-1][0],
                            distance_meters=instruction['distance'],
                            duration_seconds=instruction['time'] // 1000,
                            instructions=[instruction['text']],
                            polyline=self._encode_polyline(points)
                        )
                        segments.append(segment)

                return segments

            @staticmethod
            def _encode_polyline(coordinates: List[List[float]]) -> str:
                """Encode coordinates as polyline string"""
                import polyline
                # Convert [lon, lat] to [lat, lon]
                points = [(coord[1], coord[0]) for coord in coordinates]
                return polyline.encode(points)

            @staticmethod
            def _get_cache_key(waypoints: List[Tuple[float, float]], vehicle: str) -> str:
                """Generate cache key for route"""
                # Round to 4 decimal places (~11 meters precision)
                waypoints_str = '_'.join([f"{lat:.4f},{lon:.4f}" for lat, lon in waypoints])
                return f"route:{vehicle}:{waypoints_str}"
        ```

        ---

        ## Dynamic Rerouting

        **Problem:** Traffic changes during delivery. Driver might get stuck.

        **Solution:** Periodically recalculate route based on current location.

        ```python
        class DynamicNavigationService:
            """Real-time route updates during delivery"""

            REROUTE_CHECK_INTERVAL = 60  # Check every 60 seconds
            DEVIATION_THRESHOLD_METERS = 500  # Reroute if 500m off-route

            def __init__(self, routing_engine):
                self.routing = routing_engine

            def check_and_reroute(
                self,
                order_id: str,
                driver_lat: float,
                driver_lon: float,
                destination_lat: float,
                destination_lon: float,
                current_route: List[RouteSegment]
            ) -> Optional[List[RouteSegment]]:
                """
                Check if driver needs rerouting

                Args:
                    order_id: Order ID
                    driver_lat, driver_lon: Current driver location
                    destination_lat, destination_lon: Delivery destination
                    current_route: Current planned route

                Returns:
                    New route if rerouting needed, None otherwise
                """
                # Check if driver is on route
                deviation = self._calculate_deviation_from_route(
                    driver_lat, driver_lon,
                    current_route
                )

                if deviation > self.DEVIATION_THRESHOLD_METERS:
                    logger.info(f"Driver off-route by {deviation}m, rerouting...")

                    # Calculate new route from current location
                    new_route = self.routing.calculate_route([
                        (driver_lat, driver_lon),
                        (destination_lat, destination_lon)
                    ])

                    # Update order with new ETA
                    new_eta = sum(seg.duration_seconds for seg in new_route) // 60
                    self._update_order_eta(order_id, new_eta)

                    return new_route

                # Check if traffic has changed significantly
                current_time = datetime.utcnow()
                new_eta = self.routing.calculate_eta(
                    driver_lat, driver_lon,
                    destination_lat, destination_lon,
                    departure_time=current_time
                )

                old_eta = sum(seg.duration_seconds for seg in current_route) // 60

                if abs(new_eta - old_eta) > 5:  # 5 minute difference
                    logger.info(f"ETA changed by {new_eta - old_eta} minutes, rerouting...")
                    new_route = self.routing.calculate_route([
                        (driver_lat, driver_lon),
                        (destination_lat, destination_lon)
                    ])
                    self._update_order_eta(order_id, new_eta)
                    return new_route

                return None

            @staticmethod
            def _calculate_deviation_from_route(
                lat: float,
                lon: float,
                route: List[RouteSegment]
            ) -> float:
                """
                Calculate perpendicular distance from point to route

                Returns:
                    Distance in meters
                """
                min_distance = float('inf')

                for segment in route:
                    # Calculate distance to line segment
                    distance = _point_to_segment_distance(
                        lat, lon,
                        segment.from_lat, segment.from_lon,
                        segment.to_lat, segment.to_lon
                    )
                    min_distance = min(min_distance, distance)

                return min_distance
        ```

    === "⏱️ ETA Prediction"

        ## The Challenge

        **Problem:** Accurately predict delivery time considering:

        - Restaurant preparation time (varies by order size, time of day)
        - Driver pickup time (distance, traffic)
        - Delivery time (distance, traffic, parking difficulty)
        - Real-world delays (finding parking, elevator wait, gated communities)

        **Accuracy target:** Within 5 minutes error

        ---

        ## ML-Based ETA Prediction

        **Two-stage model:**

        1. **Preparation time:** Predict when food will be ready
        2. **Delivery time:** Predict driver travel time

        **Total ETA = Prep Time + Pickup Time + Delivery Time**

        ---

        ## Implementation

        ```python
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from datetime import datetime, timedelta
        from typing import Dict, Optional

        class ETAPredictionService:
            """ML-based ETA prediction"""

            def __init__(self, model_prep, model_delivery, routing_engine):
                self.model_prep = model_prep  # Pre-trained ML model
                self.model_delivery = model_delivery
                self.routing = routing_engine

            def predict_total_eta(
                self,
                order: Dict,
                driver: Optional[Dict] = None
            ) -> Dict:
                """
                Predict total delivery time

                Args:
                    order: Order details
                    driver: Driver details (if assigned)

                Returns:
                    {
                        'prep_time_min': int,
                        'pickup_time_min': int,
                        'delivery_time_min': int,
                        'total_eta_min': int,
                        'estimated_delivery_time': datetime,
                        'confidence': float  # 0.0-1.0
                    }
                """
                # 1. Predict preparation time
                prep_time = self._predict_prep_time(order)

                # 2. Calculate pickup time (if driver assigned)
                if driver:
                    pickup_time = self.routing.calculate_eta(
                        driver['lat'], driver['lon'],
                        order['restaurant_lat'], order['restaurant_lon']
                    )
                else:
                    # Estimate: average driver 2 miles away
                    pickup_time = 8  # minutes

                # 3. Predict delivery time
                delivery_time = self._predict_delivery_time(order, driver)

                # 4. Total ETA
                total_eta = prep_time + pickup_time + delivery_time

                # 5. Calculate confidence (based on historical accuracy)
                confidence = self._calculate_confidence(order, driver)

                # 6. Estimated delivery timestamp
                estimated_time = datetime.utcnow() + timedelta(minutes=total_eta)

                return {
                    'prep_time_min': prep_time,
                    'pickup_time_min': pickup_time,
                    'delivery_time_min': delivery_time,
                    'total_eta_min': total_eta,
                    'estimated_delivery_time': estimated_time.isoformat(),
                    'confidence': confidence
                }

            def _predict_prep_time(self, order: Dict) -> int:
                """
                Predict restaurant preparation time using ML

                Features:
                - Number of items
                - Item complexity (e.g., pizza takes longer than salad)
                - Time of day (rush hour = slower)
                - Day of week
                - Restaurant's historical prep time
                - Current restaurant load (pending orders)

                Returns:
                    Prep time in minutes
                """
                features = self._extract_prep_features(order)

                # Predict using pre-trained model
                prep_time = self.model_prep.predict([features])[0]

                # Clamp to reasonable range
                return int(max(5, min(prep_time, 60)))

            def _predict_delivery_time(self, order: Dict, driver: Optional[Dict]) -> int:
                """
                Predict delivery time from restaurant to customer

                Features:
                - Distance (miles)
                - Time of day (traffic)
                - Day of week
                - Weather (rain = slower)
                - Delivery zone (urban vs suburban)
                - Historical delivery time for this route
                - Driver experience

                Returns:
                    Delivery time in minutes
                """
                features = self._extract_delivery_features(order, driver)

                # Predict using pre-trained model
                delivery_time = self.model_delivery.predict([features])[0]

                # Clamp to reasonable range
                return int(max(5, min(delivery_time, 90)))

            def _extract_prep_features(self, order: Dict) -> np.ndarray:
                """Extract features for prep time prediction"""
                now = datetime.utcnow()

                features = [
                    len(order['items']),  # Item count
                    sum(item['quantity'] for item in order['items']),  # Total quantity
                    order['subtotal'],  # Order value (proxy for complexity)
                    now.hour,  # Hour of day
                    now.weekday(),  # Day of week (0=Monday, 6=Sunday)
                    self._is_peak_hour(now),  # 1 if peak, 0 otherwise
                    order.get('restaurant_average_prep_time', 20),  # Historical avg
                    order.get('restaurant_current_load', 5)  # Pending orders
                ]

                return np.array(features)

            def _extract_delivery_features(self, order: Dict, driver: Optional[Dict]) -> np.ndarray:
                """Extract features for delivery time prediction"""
                now = datetime.utcnow()

                # Calculate distance
                distance_miles = self._haversine_distance(
                    order['restaurant_lat'], order['restaurant_lon'],
                    order['delivery_lat'], order['delivery_lon']
                )

                features = [
                    distance_miles,  # Distance
                    now.hour,  # Hour of day
                    now.weekday(),  # Day of week
                    self._is_peak_hour(now),  # Peak traffic
                    self._is_raining(order['delivery_lat'], order['delivery_lon']),  # Weather
                    self._get_zone_type(order['delivery_lat'], order['delivery_lon']),  # Urban=1, Suburban=0
                    driver['rating'] if driver else 4.5,  # Driver rating
                    driver['total_deliveries'] if driver else 100  # Driver experience
                ]

                return np.array(features)

            def _calculate_confidence(self, order: Dict, driver: Optional[Dict]) -> float:
                """
                Calculate confidence score based on feature quality

                Returns:
                    Confidence 0.0-1.0
                """
                confidence = 1.0

                # Reduce confidence if no driver assigned yet
                if not driver:
                    confidence *= 0.8

                # Reduce confidence during peak hours (more variability)
                if self._is_peak_hour(datetime.utcnow()):
                    confidence *= 0.9

                # Reduce confidence for new restaurants (less historical data)
                if order.get('restaurant_total_orders', 1000) < 100:
                    confidence *= 0.85

                return confidence

            @staticmethod
            def _is_peak_hour(time: datetime) -> int:
                """Check if current time is peak hour"""
                hour = time.hour
                weekday = time.weekday()

                if weekday < 5:  # Weekday
                    return 1 if (11 <= hour <= 13 or 18 <= hour <= 20) else 0
                else:  # Weekend
                    return 1 if (12 <= hour <= 14 or 18 <= hour <= 20) else 0
        ```

        ---

        ## Model Training

        **Training data:**

        - Historical orders (100M+ orders)
        - Features: order details, restaurant, driver, time, location
        - Target: actual prep time, actual delivery time

        **Algorithm:** Gradient Boosting (handles non-linear relationships)

        **Training frequency:** Retrain weekly with fresh data

        ```python
        class ETAModelTrainer:
            """Train ETA prediction models"""

            def train_prep_time_model(self, training_data):
                """
                Train preparation time prediction model

                Args:
                    training_data: DataFrame with columns:
                        - item_count, total_quantity, order_value, hour, day_of_week,
                          is_peak, restaurant_avg_prep, restaurant_load
                        - actual_prep_time (target)
                """
                X = training_data.drop('actual_prep_time', axis=1)
                y = training_data['actual_prep_time']

                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    loss='huber'  # Robust to outliers
                )

                model.fit(X, y)

                # Evaluate
                from sklearn.metrics import mean_absolute_error
                predictions = model.predict(X)
                mae = mean_absolute_error(y, predictions)
                logger.info(f"Prep time model MAE: {mae:.2f} minutes")

                return model

            def train_delivery_time_model(self, training_data):
                """Train delivery time prediction model"""
                X = training_data.drop('actual_delivery_time', axis=1)
                y = training_data['actual_delivery_time']

                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    loss='huber'
                )

                model.fit(X, y)

                # Evaluate
                from sklearn.metrics import mean_absolute_error
                predictions = model.predict(X)
                mae = mean_absolute_error(y, predictions)
                logger.info(f"Delivery time model MAE: {mae:.2f} minutes")

                return model
        ```

    === "💰 Surge Pricing & Demand Forecasting"

        ## The Challenge

        **Problem:** Balance supply (drivers) and demand (orders) using dynamic pricing.

        **Goals:**

        - Incentivize drivers to go online during high demand
        - Reduce customer demand during extreme peaks
        - Maximize platform revenue
        - Maintain fairness (no price gouging)

        **Factors:**

        - Current driver availability
        - Current order volume
        - Historical demand patterns
        - Weather, events, holidays
        - Time of day

        ---

        ## Surge Pricing Algorithm

        **Formula:**

        ```
        SurgeMultiplier = max(1.0, min(MaxSurge, BaseSurge × DemandRatio × EventMultiplier))

        where:
        - DemandRatio = ActiveOrders / AvailableDrivers
        - EventMultiplier = 1.0 (normal) to 2.0 (major event)
        - MaxSurge = 3.0 (cap at 3x normal price)
        ```

        ---

        ## Implementation

        ```python
        from typing import Dict, List
        from datetime import datetime, timedelta
        import math

        class SurgePricingService:
            """Dynamic pricing based on supply/demand"""

            # Surge configuration
            BASE_SURGE_MULTIPLIER = 1.2
            MAX_SURGE_MULTIPLIER = 3.0
            MIN_SURGE_MULTIPLIER = 1.0

            # Demand/supply thresholds
            BALANCED_RATIO = 1.0  # 1 driver per 1 active order
            HIGH_DEMAND_RATIO = 2.0  # 2 orders per driver
            EXTREME_DEMAND_RATIO = 5.0  # 5 orders per driver

            def __init__(self, redis_client, analytics_service):
                self.redis = redis_client
                self.analytics = analytics_service

            def calculate_surge_multiplier(
                self,
                lat: float,
                lon: float,
                time: datetime
            ) -> Dict:
                """
                Calculate surge multiplier for location and time

                Args:
                    lat, lon: Delivery location
                    time: Order time

                Returns:
                    {
                        'multiplier': float,
                        'reason': str,
                        'supply_demand_ratio': float,
                        'expires_at': datetime
                    }
                """
                # 1. Get supply/demand in area
                geohash = self._get_geohash(lat, lon, precision=5)
                available_drivers = self._count_available_drivers(geohash)
                active_orders = self._count_active_orders(geohash)

                # Avoid division by zero
                if available_drivers == 0:
                    demand_ratio = self.EXTREME_DEMAND_RATIO
                else:
                    demand_ratio = active_orders / available_drivers

                # 2. Calculate base surge
                if demand_ratio <= self.BALANCED_RATIO:
                    base_surge = self.MIN_SURGE_MULTIPLIER
                    reason = "Normal demand"
                elif demand_ratio <= self.HIGH_DEMAND_RATIO:
                    base_surge = 1.0 + (demand_ratio - 1.0) * 0.5
                    reason = "Moderate demand"
                elif demand_ratio <= self.EXTREME_DEMAND_RATIO:
                    base_surge = 2.0 + (demand_ratio - 2.0) * 0.3
                    reason = "High demand"
                else:
                    base_surge = 2.5
                    reason = "Very high demand"

                # 3. Apply event multiplier
                event_multiplier = self._get_event_multiplier(lat, lon, time)
                if event_multiplier > 1.0:
                    reason += f" + special event"

                # 4. Apply weather multiplier
                weather_multiplier = self._get_weather_multiplier(lat, lon)
                if weather_multiplier > 1.0:
                    reason += f" + bad weather"

                # 5. Calculate final multiplier
                surge_multiplier = base_surge * event_multiplier * weather_multiplier

                # 6. Clamp to min/max
                surge_multiplier = max(
                    self.MIN_SURGE_MULTIPLIER,
                    min(surge_multiplier, self.MAX_SURGE_MULTIPLIER)
                )

                # 7. Round to nearest 0.1
                surge_multiplier = round(surge_multiplier, 1)

                # 8. Cache for 1 minute
                expires_at = time + timedelta(minutes=1)
                cache_key = f"surge:{geohash}:{time.strftime('%Y%m%d%H%M')}"
                self.redis.setex(
                    cache_key,
                    60,
                    {
                        'multiplier': surge_multiplier,
                        'reason': reason,
                        'demand_ratio': demand_ratio
                    }
                )

                return {
                    'multiplier': surge_multiplier,
                    'reason': reason,
                    'supply_demand_ratio': demand_ratio,
                    'expires_at': expires_at.isoformat()
                }

            def _count_available_drivers(self, geohash: str) -> int:
                """Count available drivers in geohash area"""
                # Get drivers in geohash cell and neighbors
                driver_count = 0
                for gh in self._get_geohash_neighbors(geohash):
                    count = self.redis.scard(f'geohash:{gh}:available')
                    driver_count += count

                return driver_count

            def _count_active_orders(self, geohash: str) -> int:
                """Count active orders in geohash area"""
                order_count = 0
                for gh in self._get_geohash_neighbors(geohash):
                    count = self.redis.scard(f'geohash:{gh}:active_orders')
                    order_count += count

                return order_count

            def _get_event_multiplier(self, lat: float, lon: float, time: datetime) -> float:
                """
                Get event-based multiplier

                Check for:
                - Major sports events (Super Bowl, World Cup)
                - Concerts, festivals
                - Holidays
                - Local events

                Returns:
                    Multiplier 1.0-2.0
                """
                # Check events database
                events = self.analytics.get_nearby_events(lat, lon, time, radius_miles=1.0)

                if not events:
                    return 1.0

                # Find highest impact event
                max_multiplier = 1.0
                for event in events:
                    # Event impact based on expected attendance
                    if event['expected_attendance'] > 50000:
                        max_multiplier = max(max_multiplier, 1.8)
                    elif event['expected_attendance'] > 10000:
                        max_multiplier = max(max_multiplier, 1.5)
                    elif event['expected_attendance'] > 1000:
                        max_multiplier = max(max_multiplier, 1.2)

                return max_multiplier

            def _get_weather_multiplier(self, lat: float, lon: float) -> float:
                """
                Get weather-based multiplier

                Bad weather:
                - Rain: 1.2x (fewer drivers, harder delivery)
                - Snow: 1.5x
                - Extreme heat/cold: 1.1x

                Returns:
                    Multiplier 1.0-1.5
                """
                weather = self.analytics.get_current_weather(lat, lon)

                if not weather:
                    return 1.0

                multiplier = 1.0

                # Rain
                if weather.get('precipitation_mm', 0) > 0:
                    if weather['precipitation_mm'] > 10:  # Heavy rain
                        multiplier = 1.5
                    elif weather['precipitation_mm'] > 2:  # Moderate rain
                        multiplier = 1.3
                    else:  # Light rain
                        multiplier = 1.2

                # Snow
                if weather.get('snow_cm', 0) > 0:
                    multiplier = max(multiplier, 1.5)

                # Temperature
                temp_f = weather.get('temperature_f', 70)
                if temp_f < 20 or temp_f > 100:
                    multiplier = max(multiplier, 1.1)

                return multiplier

            @staticmethod
            def _get_geohash(lat: float, lon: float, precision: int = 5) -> str:
                """Get geohash for location"""
                import geohash2
                return geohash2.encode(lat, lon, precision=precision)

            @staticmethod
            def _get_geohash_neighbors(geohash: str) -> List[str]:
                """Get geohash and all neighbors"""
                import geohash2
                neighbors = geohash2.neighbors(geohash)
                return [geohash] + neighbors
        ```

        ---

        ## Demand Forecasting

        **Predict demand 30-60 minutes ahead to:**

        - Proactively incentivize drivers
        - Prepare restaurants for rush
        - Adjust surge pricing preemptively

        ```python
        class DemandForecastingService:
            """Predict future order demand"""

            def __init__(self, ml_model):
                self.model = ml_model  # Pre-trained time-series model

            def forecast_demand(
                self,
                lat: float,
                lon: float,
                forecast_horizon_minutes: int = 30
            ) -> Dict:
                """
                Forecast order demand for location

                Args:
                    lat, lon: Location
                    forecast_horizon_minutes: How far ahead to predict

                Returns:
                    {
                        'predicted_orders': int,
                        'confidence_interval': (int, int),
                        'surge_risk': str  # 'low', 'medium', 'high'
                    }
                """
                # Extract features
                features = self._extract_forecast_features(lat, lon, forecast_horizon_minutes)

                # Predict
                predicted_orders = self.model.predict([features])[0]

                # Calculate confidence interval
                std_dev = predicted_orders * 0.2  # Assume 20% standard deviation
                confidence_interval = (
                    int(predicted_orders - std_dev),
                    int(predicted_orders + std_dev)
                )

                # Assess surge risk
                current_drivers = self._get_driver_forecast(lat, lon, forecast_horizon_minutes)
                demand_ratio = predicted_orders / max(current_drivers, 1)

                if demand_ratio > 3.0:
                    surge_risk = 'high'
                elif demand_ratio > 1.5:
                    surge_risk = 'medium'
                else:
                    surge_risk = 'low'

                return {
                    'predicted_orders': int(predicted_orders),
                    'confidence_interval': confidence_interval,
                    'surge_risk': surge_risk
                }

            def _extract_forecast_features(
                self,
                lat: float,
                lon: float,
                horizon_minutes: int
            ) -> np.ndarray:
                """Extract features for demand forecasting"""
                target_time = datetime.utcnow() + timedelta(minutes=horizon_minutes)

                features = [
                    target_time.hour,
                    target_time.weekday(),
                    self._is_holiday(target_time),
                    self._get_zone_density(lat, lon),  # Restaurant density
                    self._get_historical_demand(lat, lon, target_time),  # Same time last week
                    self._get_nearby_events_impact(lat, lon, target_time),
                    self._get_weather_impact(lat, lon)
                ]

                return np.array(features)
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling food delivery from 1M to 50M orders/day.

    **Scaling challenges at 50M orders/day:**

    - **Location updates:** 400K updates/sec from drivers
    - **Matching throughput:** 2,900 matches/sec during peak
    - **Storage:** 662 TB total data (638 TB orders + 24 TB location history)
    - **Real-time tracking:** 2M concurrent active orders

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **PostgreSQL writes** | ✅ Yes | Shard by order_id/user_id, write replicas, connection pooling |
    | **Redis geospatial** | ✅ Yes | Cluster mode, 50 shards by geohash prefix, 400K ops/sec achieved |
    | **Driver matching** | ✅ Yes | Batch matching (5s intervals), parallel scoring, Hungarian algorithm |
    | **Route calculations** | ✅ Yes | Self-hosted GraphHopper, route caching (5 min TTL), 50ms p95 |
    | **Location streaming** | ✅ Yes | Kafka 30 partitions by driver_id, batched writes to TimescaleDB |
    | **ETA predictions** | 🟡 Approaching | Model caching, async updates, feature precomputation |

    ---

    ## Database Sharding

    ### Orders Database

    **Sharding strategy:** Hash by order_id (consistent hashing)

    ```
    50M orders/day × 365 days = 18.25B orders/year
    18.25B × 7 KB = 127 TB/year

    Shard across 32 PostgreSQL instances
    - Each shard: 4 TB/year
    - Query routing: order_id → shard_id = hash(order_id) % 32
    ```

    **Benefits:**

    - **Linear scaling:** Add more shards as needed
    - **Isolated failures:** One shard down doesn't affect others
    - **Parallel queries:** User order history queries parallelized

    ---

    ### Location Database (TimescaleDB)

    **Partitioning:** By time (daily) + driver_id

    ```
    400K updates/sec × 100 bytes = 40 MB/sec = 3.4 TB/day

    Retention: 7 days hot data, 90 days cold (archive to S3)
    Hot data: 7 × 3.4 TB = 24 TB
    ```

    **Optimizations:**

    - **Batch inserts:** Buffer 1000 updates, insert every 5 seconds
    - **Compression:** TimescaleDB compression reduces to 30% size
    - **Indexes:** Only on (driver_id, timestamp) for queries

    ---

    ## Redis Geospatial Clustering

    **Challenge:** 2M active drivers, 400K location updates/sec

    **Solution:** Redis Cluster with geohash-based sharding

    ```
    Shard by geohash prefix (2 characters):
    - 32 × 32 = 1,024 possible shards
    - Use 50 shards (balance between granularity and complexity)
    - Each shard: 40K drivers, 8K updates/sec
    ```

    **Implementation:**

    ```python
    class RedisGeoCluster:
        """Sharded Redis for geospatial data"""

        def __init__(self, redis_clients: List[Redis]):
            self.clients = redis_clients  # 50 Redis instances
            self.num_shards = len(redis_clients)

        def get_shard(self, geohash: str) -> Redis:
            """Route to shard based on geohash prefix"""
            # Use first 2 characters of geohash
            prefix = geohash[:2]
            shard_id = hash(prefix) % self.num_shards
            return self.clients[shard_id]

        def update_location(self, driver_id: str, lat: float, lon: float):
            """Update driver location in correct shard"""
            import geohash2
            gh = geohash2.encode(lat, lon, precision=6)
            shard = self.get_shard(gh)

            # Update in Redis
            shard.geoadd('driver_locations', lon, lat, driver_id)
            shard.hset(f'driver:{driver_id}', mapping={
                'lat': lat,
                'lon': lon,
                'geohash': gh,
                'updated_at': time.time()
            })

        def get_nearby_drivers(self, lat: float, lon: float, radius_miles: float) -> List[str]:
            """Query multiple shards for nearby drivers"""
            import geohash2
            gh = geohash2.encode(lat, lon, precision=2)
            neighbors = geohash2.neighbors(gh)

            # Query all relevant shards in parallel
            from concurrent.futures import ThreadPoolExecutor
            all_drivers = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for neighbor_gh in [gh] + neighbors:
                    shard = self.get_shard(neighbor_gh)
                    future = executor.submit(
                        shard.georadius,
                        'driver_locations',
                        lon, lat,
                        radius_miles * 1609.34,  # Convert to meters
                        unit='m',
                        withdist=True
                    )
                    futures.append(future)

                for future in futures:
                    drivers = future.result()
                    all_drivers.extend(drivers)

            return all_drivers
    ```

    ---

    ## Kafka Partitioning

    **Topic:** `driver_locations`

    - **Partitions:** 30 (balance throughput and consumer parallelism)
    - **Partition key:** `driver_id` (ensures all updates for a driver go to same partition)
    - **Throughput:** 400K msgs/sec ÷ 30 = 13.3K msgs/sec per partition

    **Consumer groups:**

    - **Location processor:** 30 consumers (one per partition)
    - **Real-time tracker:** 30 consumers (push to WebSocket)
    - **Analytics:** 10 consumers (aggregate for demand forecasting)

    ---

    ## Caching Strategy

    | Cache Type | Data | TTL | Hit Rate |
    |------------|------|-----|----------|
    | **Restaurant cache** | Menu, metadata, images | 1 hour | 85% |
    | **Route cache** | Pre-calculated routes | 5 min | 60% |
    | **Active orders** | Order status, driver location | 30 sec | 95% |
    | **Driver locations** | Real-time positions | 10 sec | 99% |
    | **Surge pricing** | Multiplier by geohash | 1 min | 70% |

    **Cache invalidation:**

    - **Event-driven:** Kafka events trigger cache invalidation
    - **TTL-based:** Automatic expiration for most data
    - **Manual:** Admin tool to purge specific keys

    ---

    ## Cost Optimization

    **Monthly cost at 50M orders/day:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $25,920 (180 × m5.xlarge) |
    | **PostgreSQL RDS** | $43,200 (32 shards × db.r5.xlarge) |
    | **TimescaleDB** | $21,600 (20 nodes) |
    | **Redis Cluster** | $32,400 (50 nodes × r5.large) |
    | **Kafka Cluster** | $10,800 (30 brokers) |
    | **Elasticsearch** | $8,640 (20 nodes) |
    | **S3 Storage** | $1,500 (100 TB) |
    | **Data Transfer** | $15,000 (500 TB/month) |
    | **GraphHopper Servers** | $4,320 (30 × m5.large) |
    | **Google Maps API** | $10,000 (fallback, 2M requests) |
    | **Total** | **$173,380/month** |

    **Cost per order:** $173,380 ÷ (50M × 30) = **$0.116**

    **Revenue:** $30 order × 20% commission = **$6.00** → Healthy margin

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Matching Latency (P95)** | < 30s | > 60s |
    | **Location Update Rate** | 400K/sec | < 300K/sec (missing updates) |
    | **ETA Accuracy (MAE)** | < 5 min | > 10 min |
    | **Driver Utilization** | > 60% | < 40% (oversupply) |
    | **Order Cancellation Rate** | < 5% | > 10% |
    | **Database Connection Pool** | < 80% | > 95% |
    | **Cache Hit Rate** | > 80% | < 60% |

    **Alerting tools:**

    - **Datadog:** Real-time metrics, anomaly detection
    - **PagerDuty:** On-call rotations, escalation
    - **Sentry:** Error tracking, stack traces

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Redis Geospatial:** Fast proximity queries for driver matching (< 10ms)
    2. **Geohashing:** Efficient spatial indexing for 2M active drivers
    3. **Dispatch algorithm:** Multi-factor scoring (distance, ETA, quality, opportunity)
    4. **Self-hosted routing:** GraphHopper for cost savings ($0 vs $500K/day)
    5. **ML-based ETA:** Predict delivery time within 5 minutes accuracy
    6. **Surge pricing:** Dynamic pricing balances supply/demand
    7. **TimescaleDB:** Time-series database for location history
    8. **Event-driven architecture:** Kafka for async processing (410K write QPS)

    ---

    ## Interview Tips

    ✅ **Start with geospatial indexing** - Critical for driver matching

    ✅ **Discuss dispatch algorithm trade-offs** - Nearest driver vs optimal assignment

    ✅ **Explain ETA prediction** - ML approach with prep time + delivery time

    ✅ **Cover surge pricing** - Supply/demand balancing, fairness

    ✅ **Mention routing optimizations** - Self-hosted vs API costs

    ✅ **Real-time tracking** - WebSocket vs polling, location update frequency

    ✅ **Database sharding** - Order growth requires horizontal scaling

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to match driver in <30s?"** | Redis GEORADIUS, scoring algorithm, batch matching during peak |
    | **"How to calculate accurate ETA?"** | ML model with prep time + routing engine, traffic data, historical patterns |
    | **"How to handle surge pricing?"** | Supply/demand ratio, event/weather multipliers, max cap (3x), transparency |
    | **"How to optimize route with traffic?"** | GraphHopper + historical data, dynamic rerouting every 60s, cache routes |
    | **"How to scale location updates?"** | Kafka partitioning, batched writes, Redis Cluster, TimescaleDB compression |
    | **"What if driver goes offline mid-delivery?"** | Reassign to nearest driver, notify customer, automatic rerouting |
    | **"How to prevent fraudulent orders?"** | Payment verification, user history, location verification, ML fraud detection |

    ---

    ## Variants to Consider

    **Grocery delivery (Instacart):**

    - Shopping time estimation (ML model)
    - Item substitution workflow
    - Batch orders to same driver

    **Package delivery (Amazon):**

    - Multi-stop route optimization
    - Delivery windows (not exact time)
    - Warehouse location optimization

    **Ride-sharing (Uber/Lyft):**

    - Passenger matching (similar to driver matching)
    - Real-time pricing (similar to surge)
    - Route sharing for UberPool

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** UberEats, DoorDash, GrubHub, Postmates, Deliveroo, Swiggy, Zomato

---

*Master this problem and you'll be ready for: Instacart, Amazon Logistics, Uber, Lyft, Postmates, Grubhub*
