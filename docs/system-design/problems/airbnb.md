# Design Airbnb

Design an online marketplace for short-term accommodation rentals. Users can search for properties, book stays, manage listings, handle payments, leave reviews, and communicate with hosts/guests.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 150M users, 7M listings, 500M stays/year, global operations |
| **Key Challenges** | Geospatial search, inventory management, dynamic pricing, double-booking prevention |
| **Core Concepts** | Geohashing, distributed transactions, search ranking, payment processing, calendar management |
| **Companies** | Airbnb, Booking.com, Vrbo, Expedia, Hotels.com |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Search Listings** | Search by location, dates, guests, price | P0 (Must have) |
    | **View Listing** | Photos, amenities, reviews, availability | P0 (Must have) |
    | **Book Reservation** | Reserve property for specific dates | P0 (Must have) |
    | **Payment** | Process payments securely | P0 (Must have) |
    | **Host Listing** | Create and manage property listings | P0 (Must have) |
    | **Calendar Management** | Manage availability and bookings | P0 (Must have) |
    | **Reviews** | Rate and review stays | P1 (Should have) |
    | **Messaging** | Host-guest communication | P1 (Should have) |
    | **Cancellation** | Cancel bookings with policy enforcement | P1 (Should have) |
    | **Wishlist** | Save favorite listings | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Airbnb Experiences
    - Host insurance
    - Professional photography services
    - Background checks
    - Property damage protection
    - Multi-currency complexity (assume USD)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Bookings must always work |
    | **Search Latency** | < 500ms | Fast search results critical for UX |
    | **Booking Consistency** | Strong consistency | No double-bookings allowed |
    | **Payment Reliability** | 99.99% success | Money transactions must be reliable |
    | **Scalability** | Handle 10K searches/sec | Peak travel season traffic |
    | **Data Durability** | No data loss | Bookings and payments are critical |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total users: 150 million
    Daily Active Users (DAU): 5 million
    Total listings: 7 million
    Bookings per year: 500 million

    Daily metrics:
    - Searches: 5M × 5 = 25M searches/day
    - Listing views: 5M × 10 = 50M views/day
    - Bookings: 500M / 365 = 1.37M bookings/day
    - Reviews: 1M reviews/day
    - Messages: 5M messages/day

    QPS calculations:
    - Search: 25M / 86,400 = 290 QPS (peak: 1,000 QPS)
    - Listing views: 50M / 86,400 = 580 QPS
    - Bookings: 1.37M / 86,400 = 16 bookings/sec
    - Reviews: 1M / 86,400 = 12 reviews/sec
    ```

    ### Storage Estimates

    ```
    Listings:
    - Listing data: 7M × 100 KB = 700 GB
    - Photos (20 per listing): 7M × 20 × 500 KB = 70 TB

    Bookings:
    - 500M bookings/year × 5 KB = 2.5 TB/year
    - For 5 years: 12.5 TB

    Reviews:
    - 365M reviews/year × 2 KB = 730 GB/year
    - For 5 years: 3.65 TB

    Users:
    - 150M users × 10 KB = 1.5 TB

    Messages:
    - 1.8B messages/year × 1 KB = 1.8 TB/year
    - For 1 year: 1.8 TB

    Total: 700 GB + 70 TB + 12.5 TB + 3.65 TB + 1.5 TB + 1.8 TB ≈ 90 TB
    ```

    ### Bandwidth Estimates

    ```
    Search (read):
    - 290 QPS × 50 KB = 14.5 MB/sec ≈ 116 Mbps

    Listing views (read):
    - 580 QPS × 100 KB = 58 MB/sec ≈ 464 Mbps
    - Photos via CDN: 580 QPS × 10 photos × 500 KB = 2.9 GB/sec ≈ 23 Gbps

    Booking (write):
    - 16 bookings/sec × 5 KB = 80 KB/sec

    Total read: ~23.5 Gbps (mostly CDN for photos)
    Total write: ~1 Mbps
    ```

    ### Memory Estimates (Caching)

    ```
    Hot listings (top 10%):
    - 700K listings × 100 KB = 70 GB

    Search results cache:
    - 1M unique searches × 100 KB = 100 GB

    User sessions:
    - 500K concurrent × 10 KB = 5 GB

    Availability calendar cache:
    - 1M active listings × 10 KB = 10 GB

    Total cache: 70 GB + 100 GB + 5 GB + 10 GB = 185 GB
    ```

    ---

    ## Key Assumptions

    1. Average booking: 3 nights, 2 guests
    2. Search conversion rate: 5% (5% of searches lead to booking)
    3. 80/20 rule: 20% of listings generate 80% of bookings
    4. Average 365-day availability window
    5. Peak season: Summer (2x average traffic)
    6. Average listing has 20 photos

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Geospatial search first** - Location is primary search criteria
    2. **Strong consistency for bookings** - Prevent double-booking
    3. **Eventual consistency for reads** - Search results can be slightly stale
    4. **Idempotent payments** - Handle payment retries safely
    5. **Calendar-based availability** - Efficient date range queries

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile Apps]
            Web[Web Browser]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Rate Limiting]
        end

        subgraph "Service Layer"
            Search[Search Service<br/>Geospatial]
            Listing[Listing Service]
            Booking[Booking Service<br/>Transactional]
            Payment[Payment Service<br/>Stripe Integration]
            Review[Review Service]
            Message[Messaging Service]
            Notification[Notification Service]
        end

        subgraph "Background Processing"
            PriceOptimizer[Dynamic Pricing<br/>ML Model]
            SearchIndexer[Search Indexer<br/>Elasticsearch]
            EmailWorker[Email Worker]
        end

        subgraph "Caching Layer"
            Redis_Search[Redis<br/>Search Results]
            Redis_Listing[Redis<br/>Hot Listings]
            Redis_Calendar[Redis<br/>Availability]
        end

        subgraph "Storage Layer"
            ListingDB[(Listing DB<br/>PostgreSQL)]
            BookingDB[(Booking DB<br/>PostgreSQL<br/>ACID)]
            ReviewDB[(Review DB<br/>Cassandra)]
            MessageDB[(Message DB<br/>Cassandra)]
            UserDB[(User DB<br/>PostgreSQL)]
            SearchIndex[(Elasticsearch<br/>Geospatial Search)]
        end

        subgraph "External Services"
            Stripe[Stripe<br/>Payment Gateway]
            S3[S3<br/>Photo Storage]
            CDN[CloudFront<br/>CDN]
        end

        Mobile --> Gateway
        Web --> Gateway

        Gateway --> Search
        Gateway --> Listing
        Gateway --> Booking
        Gateway --> Payment
        Gateway --> Review
        Gateway --> Message

        Search --> SearchIndex
        Search --> Redis_Search

        Listing --> ListingDB
        Listing --> Redis_Listing
        Listing --> S3

        Booking --> BookingDB
        Booking --> Redis_Calendar
        Booking --> Payment

        Payment --> Stripe
        Payment --> Notification

        Review --> ReviewDB
        Message --> MessageDB

        SearchIndexer --> SearchIndex
        SearchIndexer --> ListingDB

        PriceOptimizer --> ListingDB

        S3 --> CDN
        CDN --> Mobile
        CDN --> Web

        style Gateway fill:#e1f5ff
        style BookingDB fill:#ffe1e1
        style SearchIndex fill:#fff4e1
        style Stripe fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Technology | Why This? | Alternative |
    |-----------|-----------|-----------|-------------|
    | **Geospatial Search** | Elasticsearch + Geohash | Fast radius search, ranking, filters | PostGIS (slower), custom (complex) |
    | **Booking DB** | PostgreSQL | ACID transactions, prevent double-booking | NoSQL (no transactions), custom locking (error-prone) |
    | **Listing DB** | PostgreSQL | Complex queries, joins, consistency | NoSQL (harder to query), sharded SQL (premature) |
    | **Reviews** | Cassandra | High write throughput, append-only | PostgreSQL (scaling issues), MongoDB (consistency) |
    | **Messages** | Cassandra | Time-series data, high throughput | PostgreSQL (scaling), MongoDB (consistency) |
    | **Photo Storage** | S3 + CloudFront | Scalable, durable, fast delivery via CDN | Database (expensive), custom storage (complex) |
    | **Payment** | Stripe | PCI compliance, reliability, ease of use | Custom (complex), Braintree (similar) |
    | **Cache** | Redis | Fast lookups, geospatial support | Memcached (no geospatial), no cache (slow) |

    ---

    ## API Design

    ### 1. Search Listings

    **Request:**
    ```http
    GET /api/v1/search
      ?location=37.7749,-122.4194
      &radius=10km
      &checkin=2026-03-15
      &checkout=2026-03-18
      &guests=2
      &min_price=100
      &max_price=300
      &amenities=wifi,pool
      &property_type=entire_home
      &limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "listings": [
        {
          "listing_id": "L12345",
          "title": "Cozy Downtown Apartment",
          "location": {
            "lat": 37.7849,
            "lng": -122.4094,
            "city": "San Francisco",
            "neighborhood": "SOMA"
          },
          "price_per_night": 150,
          "total_price": 450,
          "rating": 4.8,
          "review_count": 234,
          "property_type": "Entire home",
          "bedrooms": 2,
          "beds": 2,
          "bathrooms": 1,
          "max_guests": 4,
          "photos": [
            "https://cdn.airbnb.com/photos/L12345_1.jpg"
          ],
          "is_superhost": true,
          "instant_book": true
        }
      ],
      "total_results": 478,
      "next_cursor": "eyJvZmZzZXQiOjIwfQ=="
    }
    ```

    ---

    ### 2. Get Listing Details

    **Request:**
    ```http
    GET /api/v1/listings/L12345
      ?checkin=2026-03-15
      &checkout=2026-03-18
      &guests=2
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "listing_id": "L12345",
      "title": "Cozy Downtown Apartment",
      "description": "Beautiful apartment with stunning views...",
      "host": {
        "host_id": "H789",
        "name": "John Doe",
        "joined": "2020-01-15",
        "is_superhost": true,
        "response_rate": 98,
        "response_time": "within an hour"
      },
      "location": {...},
      "pricing": {
        "price_per_night": 150,
        "cleaning_fee": 50,
        "service_fee": 45,
        "total": 545
      },
      "amenities": ["WiFi", "Kitchen", "Pool", "Parking"],
      "house_rules": "No smoking, No pets",
      "cancellation_policy": "Flexible",
      "availability": {
        "available_dates": [...],
        "min_nights": 2,
        "max_nights": 14
      },
      "reviews_summary": {
        "rating": 4.8,
        "count": 234,
        "cleanliness": 4.9,
        "accuracy": 4.8,
        "communication": 4.9
      }
    }
    ```

    ---

    ### 3. Create Booking

    **Request:**
    ```http
    POST /api/v1/bookings
    Content-Type: application/json
    Authorization: Bearer <token>
    Idempotency-Key: uuid-client-generated

    {
      "listing_id": "L12345",
      "checkin_date": "2026-03-15",
      "checkout_date": "2026-03-18",
      "guests": 2,
      "payment_method_id": "pm_stripe_token",
      "guest_details": {
        "first_name": "Jane",
        "last_name": "Smith",
        "phone": "+1234567890"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "booking_id": "B98765",
      "status": "confirmed",
      "listing_id": "L12345",
      "checkin_date": "2026-03-15",
      "checkout_date": "2026-03-18",
      "guests": 2,
      "total_amount": 545,
      "currency": "USD",
      "confirmation_code": "ABC123XYZ",
      "created_at": "2026-01-29T10:30:00Z"
    }
    ```

    ---

    ### 4. Get Availability Calendar

    **Request:**
    ```http
    GET /api/v1/listings/L12345/calendar
      ?start_date=2026-03-01
      &end_date=2026-04-01
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "listing_id": "L12345",
      "calendar": [
        {
          "date": "2026-03-15",
          "available": true,
          "price": 150,
          "min_nights": 2
        },
        {
          "date": "2026-03-16",
          "available": false,
          "blocked_by": "booking"
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Listings (PostgreSQL)

    ```sql
    CREATE TABLE listings (
        listing_id BIGINT PRIMARY KEY,
        host_id BIGINT REFERENCES users(user_id),
        title VARCHAR(200),
        description TEXT,
        property_type VARCHAR(50),
        bedrooms INT,
        beds INT,
        bathrooms DECIMAL(3,1),
        max_guests INT,
        price_per_night DECIMAL(10,2),
        cleaning_fee DECIMAL(10,2),
        location_lat DECIMAL(10,8),
        location_lng DECIMAL(11,8),
        city VARCHAR(100),
        neighborhood VARCHAR(100),
        amenities TEXT[],
        house_rules TEXT,
        cancellation_policy VARCHAR(50),
        min_nights INT DEFAULT 1,
        max_nights INT DEFAULT 365,
        instant_book BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_host_id (host_id),
        INDEX idx_location (location_lat, location_lng),
        INDEX idx_city (city)
    );
    ```

    ---

    ### Bookings (PostgreSQL)

    ```sql
    CREATE TABLE bookings (
        booking_id BIGINT PRIMARY KEY,
        listing_id BIGINT REFERENCES listings(listing_id),
        guest_id BIGINT REFERENCES users(user_id),
        checkin_date DATE NOT NULL,
        checkout_date DATE NOT NULL,
        guests INT,
        status VARCHAR(20), -- pending, confirmed, cancelled, completed
        total_amount DECIMAL(10,2),
        payment_status VARCHAR(20),
        payment_intent_id VARCHAR(100),
        confirmation_code VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_listing_dates (listing_id, checkin_date, checkout_date),
        INDEX idx_guest_id (guest_id),
        INDEX idx_status (status),
        CONSTRAINT no_overlap EXCLUDE USING gist (
            listing_id WITH =,
            daterange(checkin_date, checkout_date) WITH &&
        ) WHERE (status IN ('confirmed', 'pending'))
    );
    ```

    **Key constraint:** PostgreSQL exclusion constraint prevents overlapping date ranges for same listing.

    ---

    ### Availability Calendar (PostgreSQL)

    ```sql
    CREATE TABLE availability (
        listing_id BIGINT REFERENCES listings(listing_id),
        date DATE,
        available BOOLEAN DEFAULT TRUE,
        price DECIMAL(10,2),
        min_nights INT DEFAULT 1,
        PRIMARY KEY (listing_id, date),
        INDEX idx_listing_date (listing_id, date)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Booking Flow (Critical Path)

    ```mermaid
    sequenceDiagram
        participant Guest
        participant API
        participant Booking_Service
        participant Listing_DB
        participant Booking_DB
        participant Payment
        participant Notification

        Guest->>API: Create booking request
        API->>Booking_Service: Process booking
        Booking_Service->>Booking_Service: Check idempotency key

        Booking_Service->>Listing_DB: Get listing details
        Listing_DB-->>Booking_Service: Listing info

        Booking_Service->>Booking_DB: BEGIN TRANSACTION
        Booking_Service->>Booking_DB: Check date availability<br/>(with row lock)

        alt Dates available
            Booking_DB-->>Booking_Service: Available
            Booking_Service->>Payment: Charge payment
            Payment-->>Booking_Service: Payment success

            Booking_Service->>Booking_DB: INSERT booking
            Booking_Service->>Booking_DB: COMMIT

            Booking_Service->>Notification: Send confirmation email
            Booking_Service-->>Guest: 201 Created
        else Dates unavailable
            Booking_DB-->>Booking_Service: Conflict
            Booking_Service->>Booking_DB: ROLLBACK
            Booking_Service-->>Guest: 409 Conflict
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics:

    ### 1. Geospatial Search
    - **Geohashing** for location indexing
    - **Elasticsearch geo_point** for radius queries
    - **Ranking algorithm:** distance + price + rating + availability
    - **Filters:** property type, amenities, price range

    ### 2. Double-Booking Prevention
    - **PostgreSQL exclusion constraint** on date ranges
    - **Row-level locking** during booking
    - **Database transaction** (ACID guarantees)
    - **Idempotency keys** for retry safety

    ### 3. Dynamic Pricing
    - **ML model** predicts optimal price
    - **Factors:** demand, seasonality, events, competitor prices
    - **Real-time updates** to availability calendar
    - **Host override** allowed

    ### 4. Availability Calendar
    - **365-day rolling window**
    - **Efficient date range queries**
    - **Blocked dates:** bookings, host blocks, maintenance

=== "⚡ Step 4: Scale & Optimize"

    ## Performance Optimization

    - **Cache hot listings** (top 10% in Redis)
    - **CDN for photos** (99% of bandwidth)
    - **Search result caching** (5-minute TTL)
    - **Database sharding** (by geographic region)
    - **Read replicas** for listing reads

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Elasticsearch for geospatial search** - Fast radius queries
    2. **PostgreSQL exclusion constraints** - Prevent double-booking
    3. **ACID transactions** - Strong consistency for bookings
    4. **Stripe for payments** - PCI compliance, reliability
    5. **Redis caching** - Hot listings, search results
    6. **S3 + CDN** - Scalable photo delivery

    ## Interview Tips

    ✅ **Emphasize double-booking prevention** - This is critical
    ✅ **Geospatial search complexity** - Explain geohashing
    ✅ **Payment idempotency** - Handle retries safely
    ✅ **Calendar queries** - Efficient date range lookups

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Airbnb, Booking.com, Vrbo, Expedia, Hotels.com

---

*Master this problem and you'll understand booking systems, geospatial search, and distributed transactions.*
