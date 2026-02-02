# Design Ticket Booking System (BookMyShow)

An online ticket booking platform for movies, events, and concerts where users can search shows, select seats, complete payment, and receive booking confirmations with strong inventory consistency.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M bookings/month, 100K active events, 10M seats, 5M DAU |
| **Key Challenges** | Seat locking, race conditions, double booking prevention, high concurrency, payment integration |
| **Core Concepts** | Pessimistic/optimistic locking, distributed locks, idempotency, ACID transactions, inventory management |
| **Companies** | BookMyShow, Ticketmaster, Eventbrite, StubHub, Fandango, Paytm Insider |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Search Events** | Search movies/events by location, date, venue | P0 (Must have) |
    | **View Seats** | Display seat layout with availability status | P0 (Must have) |
    | **Select & Lock Seats** | Reserve seats temporarily (10-15 min) | P0 (Must have) |
    | **Make Payment** | Complete booking via payment gateway | P0 (Must have) |
    | **Booking Confirmation** | Generate ticket with QR code | P0 (Must have) |
    | **View Bookings** | User booking history | P1 (Should have) |
    | **Cancel Booking** | Cancel tickets with refund | P1 (Should have) |
    | **Dynamic Pricing** | Surge pricing for popular shows | P2 (Nice to have) |
    | **Notifications** | SMS/email booking confirmations | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Food/beverage ordering
    - Loyalty programs
    - Review/rating system
    - Social features (watch parties)
    - Content streaming

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Consistency** | Strong consistency for inventory | Zero double-booking tolerance, money is involved |
    | **Availability** | 99.99% uptime | Users expect 24/7 access to book tickets |
    | **Latency (Seat Lock)** | < 500ms p95 | Critical for user experience during booking |
    | **Latency (Search)** | < 1s p95 | Fast discovery of shows |
    | **Throughput** | 10K concurrent bookings | Handle peak loads (Friday evening releases) |
    | **Data Integrity** | ACID compliance | No lost bookings, accurate inventory |
    | **No Double Booking** | 100% guarantee | Business-critical requirement |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 5M
    Monthly Active Users (MAU): 20M

    Bookings:
    - Monthly bookings: 50M
    - Daily bookings: 50M / 30 = ~1.67M bookings/day
    - Booking QPS: 1.67M / 86,400 = ~19.3 bookings/sec
    - Peak QPS: 10x average = ~193 bookings/sec (Friday evening, new releases)

    Seat locks:
    - Lock to booking conversion: 30% (many users abandon)
    - Seat lock attempts: 19.3 / 0.3 = ~64 locks/sec
    - Peak lock QPS: ~640 locks/sec

    Search queries:
    - Searches per DAU: 5 searches/day
    - Daily searches: 5M × 5 = 25M searches/day
    - Search QPS: 25M / 86,400 = ~289 req/sec

    Total Read QPS: ~350 (search + seat availability + booking history)
    Total Write QPS: ~260 (seat locks + bookings + cancellations)
    Read/Write ratio: 1.3:1 (fairly balanced)
    ```

    ### Storage Estimates

    ```
    Events data:
    - Active events: 100K
    - Event metadata: 5 KB (name, venue, time, pricing tiers, images)
    - Total: 100K × 5 KB = 500 MB (negligible)

    Shows (screenings/performances):
    - Shows per event: 20 (different dates/times)
    - Total shows: 100K × 20 = 2M shows
    - Show metadata: 2 KB
    - Total: 2M × 2 KB = 4 GB

    Seats:
    - Total seats: 10M (across all active shows)
    - Seat data: 200 bytes (seat_id, show_id, status, price, row, column)
    - Total: 10M × 200 bytes = 2 GB

    Bookings (10 years):
    - Total bookings: 50M/month × 12 × 10 = 6B bookings
    - Booking record: 500 bytes (booking_id, user_id, show_id, seats, payment_info, timestamps)
    - Total: 6B × 500 bytes = 3 TB

    Users:
    - Total users: 20M × 5 KB = 100 GB

    Total storage: 500 MB + 4 GB + 2 GB + 3 TB + 100 GB ≈ 3.1 TB
    (With indexes and replication: ~10 TB)
    ```

    ### Bandwidth Estimates

    ```
    Booking ingress:
    - 19.3 bookings/sec × 500 bytes = 9.65 KB/sec ≈ 77 Kbps (minimal)

    Search egress:
    - 289 searches/sec × 50 events × 5 KB = 72.25 MB/sec ≈ 578 Mbps

    Seat availability queries:
    - 100 req/sec × 500 seats × 200 bytes = 10 MB/sec ≈ 80 Mbps

    Total ingress: ~100 Kbps (very write-light)
    Total egress: ~658 Mbps (read-heavy during peak)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot events (trending movies/events):
    - Top 1,000 events × 5 KB = 5 MB
    - Shows for these events: 20K × 2 KB = 40 MB
    - Seat availability cache: 200K seats × 200 bytes = 40 MB

    Distributed locks:
    - Active seat locks: 10,000 concurrent × 100 bytes = 1 MB

    User sessions:
    - 100K concurrent users × 10 KB = 1 GB

    Payment idempotency cache:
    - 1K active payments × 1 KB = 1 MB

    Total cache: 5 MB + 40 MB + 40 MB + 1 MB + 1 GB + 1 MB ≈ 1.1 GB
    ```

    ---

    ## Key Assumptions

    1. Average 2.5 tickets per booking (family/group bookings)
    2. 70% of seat locks expire (users don't complete payment)
    3. Peak load is 10x average (Friday evening releases, popular events)
    4. Strong consistency required (financial transactions involved)
    5. Seat lock duration: 10 minutes (industry standard)
    6. Average venue capacity: 200-500 seats

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Strong consistency:** ACID transactions for seat booking (no double booking)
    2. **Pessimistic locking:** Lock seats during booking flow
    3. **Idempotency:** Ensure payment requests processed exactly once
    4. **Graceful degradation:** Show "limited availability" vs hard failures
    5. **Timeout handling:** Auto-release expired locks

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets]
            LB[Load Balancer<br/>NGINX]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Rate limiting, auth]
        end

        subgraph "Service Layer"
            Search_Service[Search Service<br/>Events & shows]
            Inventory_Service[Inventory Service<br/>Seat availability]
            Booking_Service[Booking Service<br/>Seat lock & booking]
            Payment_Service[Payment Service<br/>Payment processing]
            Notification_Service[Notification Service<br/>Email/SMS]
            User_Service[User Service<br/>User management]
        end

        subgraph "Locking & Queue"
            Redis_Lock[Redis<br/>Distributed locks]
            Lock_Worker[Lock Expiry Worker<br/>Release expired locks]
            Payment_Queue[Kafka<br/>Payment events]
        end

        subgraph "Caching"
            Redis_Cache[Redis<br/>Event cache]
            Redis_Inventory[Redis<br/>Seat availability]
            Redis_Session[Redis<br/>User sessions]
        end

        subgraph "Storage"
            Event_DB[(Event DB<br/>PostgreSQL<br/>Events & Shows)]
            Inventory_DB[(Inventory DB<br/>PostgreSQL<br/>Seats & locks)]
            Booking_DB[(Booking DB<br/>PostgreSQL<br/>Bookings)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Search_DB[(Elasticsearch<br/>Event search)]
        end

        subgraph "External Services"
            Payment_Gateway[Payment Gateway<br/>Stripe/Razorpay]
            Email_Service[Email Service<br/>SendGrid]
            SMS_Service[SMS Service<br/>Twilio]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        LB --> Gateway

        Gateway --> Search_Service
        Gateway --> Inventory_Service
        Gateway --> Booking_Service
        Gateway --> Payment_Service
        Gateway --> User_Service

        Search_Service --> Redis_Cache
        Search_Service --> Event_DB
        Search_Service --> Search_DB

        Inventory_Service --> Redis_Inventory
        Inventory_Service --> Inventory_DB

        Booking_Service --> Redis_Lock
        Booking_Service --> Inventory_DB
        Booking_Service --> Booking_DB
        Booking_Service --> Payment_Queue

        Payment_Service --> Payment_Gateway
        Payment_Service --> Booking_DB
        Payment_Service --> Payment_Queue

        Payment_Queue --> Notification_Service

        Notification_Service --> Email_Service
        Notification_Service --> SMS_Service

        Lock_Worker --> Redis_Lock
        Lock_Worker --> Inventory_DB

        User_Service --> Redis_Session
        User_Service --> User_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Gateway fill:#fff4e1
        style Redis_Lock fill:#ffe1e1
        style Redis_Cache fill:#fff4e1
        style Redis_Inventory fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Event_DB fill:#ffe1e1
        style Inventory_DB fill:#ffe1e1
        style Booking_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Search_DB fill:#e8eaf6
        style Payment_Gateway fill:#f3e5f5
        style Payment_Queue fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis (Distributed Locks)** | Prevent race conditions, coordinate seat locking across servers | Database row locks (doesn't scale), Zookeeper (overkill for locks) |
    | **PostgreSQL (Inventory)** | ACID transactions, strong consistency for seat booking | MongoDB (weak consistency), Cassandra (eventual consistency unacceptable) |
    | **Kafka (Payment Events)** | Reliable async processing, retry failed payments, audit trail | Direct API calls (no retry), RabbitMQ (less throughput) |
    | **Elasticsearch** | Fast fuzzy search, filters (location, date, price), auto-complete | SQL LIKE queries (too slow), custom search (complex) |
    | **Redis (Cache)** | Sub-10ms reads for hot events, seat availability | No cache (DB overload), Memcached (limited features) |
    | **API Gateway** | Rate limiting, authentication, request routing | Implement in each service (duplication), no gateway (security risk) |

    **Key Trade-off:** We chose **consistency over availability** for seat booking. During failures, we reject bookings rather than risk double-booking.

    ---

    ## API Design

    ### 1. Search Events

    **Request:**
    ```http
    GET /api/v1/events/search?q=avengers&city=bangalore&date=2026-02-15&type=movie
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "events": [
        {
          "event_id": "evt_789012",
          "title": "Avengers: Endgame",
          "type": "movie",
          "genre": ["Action", "Sci-Fi"],
          "duration_minutes": 181,
          "rating": "PG-13",
          "poster_url": "https://cdn.bookmyshow.com/posters/evt_789012.jpg",
          "description": "After the devastating events...",
          "languages": ["English", "Hindi", "Tamil"],
          "theaters": [
            {
              "theater_id": "thr_123",
              "theater_name": "PVR Koramangala",
              "location": "Bangalore, Karnataka",
              "shows": [
                {
                  "show_id": "shw_456",
                  "start_time": "2026-02-15T14:30:00Z",
                  "end_time": "2026-02-15T17:31:00Z",
                  "language": "English",
                  "format": "2D",
                  "available_seats": 87,
                  "total_seats": 200,
                  "price_range": {
                    "min": 200,
                    "max": 500
                  }
                }
              ]
            }
          ]
        }
      ],
      "total_results": 1,
      "page": 1
    }
    ```

    **Design Notes:**

    - Return theaters and shows together (reduce round trips)
    - Include seat availability (cached, updated every 30s)
    - Price range preview (actual prices on seat selection)

    ---

    ### 2. Get Seat Layout

    **Request:**
    ```http
    GET /api/v1/shows/shw_456/seats
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "show_id": "shw_456",
      "theater_name": "PVR Koramangala",
      "screen": "Screen 3",
      "total_seats": 200,
      "available_seats": 87,
      "layout": {
        "rows": [
          {
            "row": "A",
            "seats": [
              {
                "seat_id": "seat_a1",
                "seat_number": "A1",
                "status": "available",
                "type": "premium",
                "price": 500
              },
              {
                "seat_id": "seat_a2",
                "seat_number": "A2",
                "status": "booked",
                "type": "premium",
                "price": 500
              },
              {
                "seat_id": "seat_a3",
                "seat_number": "A3",
                "status": "locked",
                "type": "premium",
                "price": 500,
                "locked_until": "2026-02-15T14:45:00Z"
              }
            ]
          }
        ]
      },
      "seat_types": {
        "premium": { "price": 500, "color": "#FFD700" },
        "standard": { "price": 300, "color": "#87CEEB" },
        "economy": { "price": 200, "color": "#90EE90" }
      }
    }
    ```

    **Design Notes:**

    - Include seat status (available, booked, locked)
    - Show lock expiry for locked seats (for UI countdown)
    - Seat type and pricing information

    ---

    ### 3. Lock Seats

    **Request:**
    ```http
    POST /api/v1/bookings/lock
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "show_id": "shw_456",
      "seat_ids": ["seat_a1", "seat_a4", "seat_a5"],
      "lock_duration_seconds": 600
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "lock_id": "lock_abc123",
      "show_id": "shw_456",
      "seats": [
        {
          "seat_id": "seat_a1",
          "seat_number": "A1",
          "price": 500
        },
        {
          "seat_id": "seat_a4",
          "seat_number": "A4",
          "price": 500
        },
        {
          "seat_id": "seat_a5",
          "seat_number": "A5",
          "price": 500
        }
      ],
      "total_price": 1500,
      "locked_until": "2026-02-15T14:45:00Z",
      "expires_in_seconds": 600
    }
    ```

    **Error Response (Conflict):**
    ```http
    HTTP/1.1 409 Conflict
    Content-Type: application/json

    {
      "error": "seat_unavailable",
      "message": "Some seats are no longer available",
      "unavailable_seats": ["seat_a1"],
      "available_alternatives": [
        {
          "seat_id": "seat_a6",
          "seat_number": "A6",
          "price": 500
        }
      ]
    }
    ```

    **Design Notes:**

    - Atomic operation (all seats locked or none)
    - Return lock ID for payment reference
    - Include expiry time for client-side countdown timer
    - Suggest alternatives on conflict

    ---

    ### 4. Complete Booking

    **Request:**
    ```http
    POST /api/v1/bookings/complete
    Content-Type: application/json
    Authorization: Bearer <token>
    Idempotency-Key: idem_xyz789

    {
      "lock_id": "lock_abc123",
      "payment_method": "card",
      "payment_details": {
        "token": "pm_1234567890",
        "amount": 1500
      },
      "contact": {
        "email": "user@example.com",
        "phone": "+919876543210"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "booking_id": "bkg_987654",
      "status": "confirmed",
      "show": {
        "event_name": "Avengers: Endgame",
        "theater_name": "PVR Koramangala",
        "start_time": "2026-02-15T14:30:00Z"
      },
      "seats": ["A1", "A4", "A5"],
      "total_price": 1500,
      "payment": {
        "payment_id": "pay_111222",
        "status": "success",
        "amount": 1500,
        "method": "card"
      },
      "qr_code": "https://cdn.bookmyshow.com/qr/bkg_987654.png",
      "ticket_url": "https://bookmyshow.com/tickets/bkg_987654",
      "booked_at": "2026-02-15T14:35:00Z"
    }
    ```

    **Design Notes:**

    - Idempotency key required (prevent duplicate charges)
    - Return immediately after payment (async notification)
    - Include QR code for venue entry
    - Generate PDF ticket (sent via email)

    ---

    ## Database Schema

    ### Events & Shows (PostgreSQL)

    ```sql
    -- Events table
    CREATE TABLE events (
        event_id VARCHAR(50) PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL, -- movie, concert, sports, theater
        genre VARCHAR(100)[],
        description TEXT,
        duration_minutes INT,
        rating VARCHAR(10),
        language VARCHAR(50)[],
        poster_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_type_language (type, language),
        INDEX idx_title (title)
    );

    -- Theaters table
    CREATE TABLE theaters (
        theater_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        city VARCHAR(100) NOT NULL,
        address TEXT,
        latitude DECIMAL(10, 8),
        longitude DECIMAL(11, 8),
        total_screens INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_city (city),
        INDEX idx_location (latitude, longitude)
    );

    -- Shows (screenings/performances)
    CREATE TABLE shows (
        show_id VARCHAR(50) PRIMARY KEY,
        event_id VARCHAR(50) REFERENCES events(event_id),
        theater_id VARCHAR(50) REFERENCES theaters(theater_id),
        screen_number INT,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        language VARCHAR(50),
        format VARCHAR(50), -- 2D, 3D, IMAX, etc.
        total_seats INT NOT NULL,
        available_seats INT NOT NULL,
        status VARCHAR(50) DEFAULT 'active', -- active, sold_out, cancelled
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_event_theater_time (event_id, theater_id, start_time),
        INDEX idx_start_time (start_time),
        CONSTRAINT chk_seats CHECK (available_seats >= 0 AND available_seats <= total_seats)
    );
    ```

    ---

    ### Inventory & Seats (PostgreSQL)

    ```sql
    -- Seats table
    CREATE TABLE seats (
        seat_id VARCHAR(50) PRIMARY KEY,
        show_id VARCHAR(50) REFERENCES shows(show_id),
        seat_number VARCHAR(10) NOT NULL,
        row_name VARCHAR(5) NOT NULL,
        seat_type VARCHAR(50) NOT NULL, -- premium, standard, economy
        price DECIMAL(10, 2) NOT NULL,
        status VARCHAR(50) DEFAULT 'available', -- available, locked, booked
        locked_by VARCHAR(50), -- lock_id
        locked_until TIMESTAMP,
        booking_id VARCHAR(50),
        version INT DEFAULT 0, -- Optimistic locking
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_show_status (show_id, status),
        INDEX idx_lock (locked_by, locked_until),
        CONSTRAINT chk_status CHECK (status IN ('available', 'locked', 'booked'))
    );

    -- Seat locks (distributed lock tracking)
    CREATE TABLE seat_locks (
        lock_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        show_id VARCHAR(50) REFERENCES shows(show_id),
        seat_ids VARCHAR(50)[],
        total_price DECIMAL(10, 2),
        status VARCHAR(50) DEFAULT 'active', -- active, completed, expired
        locked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        INDEX idx_user_status (user_id, status),
        INDEX idx_expires (expires_at, status)
    );
    ```

    **Why PostgreSQL:**

    - **ACID transactions:** Critical for seat booking (no double booking)
    - **Row-level locking:** Pessimistic locking for seat updates
    - **Constraints:** Database-level validation (available_seats >= 0)
    - **Strong consistency:** Read-after-write consistency guaranteed

    ---

    ### Bookings (PostgreSQL)

    ```sql
    -- Bookings table
    CREATE TABLE bookings (
        booking_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        show_id VARCHAR(50) REFERENCES shows(show_id),
        seat_ids VARCHAR(50)[],
        seat_numbers VARCHAR(10)[],
        total_price DECIMAL(10, 2) NOT NULL,
        status VARCHAR(50) DEFAULT 'confirmed', -- confirmed, cancelled, refunded
        payment_id VARCHAR(50),
        payment_status VARCHAR(50),
        qr_code_url TEXT,
        booked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        cancelled_at TIMESTAMP,
        INDEX idx_user_booking (user_id, booked_at),
        INDEX idx_show_booking (show_id),
        INDEX idx_payment (payment_id)
    ) PARTITION BY RANGE (booked_at);

    -- Partition by month for efficient queries
    CREATE TABLE bookings_2026_01 PARTITION OF bookings
        FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

    CREATE TABLE bookings_2026_02 PARTITION OF bookings
        FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
    -- ... more partitions

    -- Payment transactions (audit trail)
    CREATE TABLE payment_transactions (
        transaction_id VARCHAR(50) PRIMARY KEY,
        booking_id VARCHAR(50) REFERENCES bookings(booking_id),
        payment_gateway VARCHAR(50),
        gateway_transaction_id VARCHAR(255),
        amount DECIMAL(10, 2),
        currency VARCHAR(10) DEFAULT 'INR',
        status VARCHAR(50), -- pending, success, failed, refunded
        idempotency_key VARCHAR(255) UNIQUE,
        payment_method VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_booking (booking_id),
        INDEX idx_idempotency (idempotency_key)
    );
    ```

    ---

    ### Users (PostgreSQL Sharded)

    ```sql
    -- Users table (sharded by user_id)
    CREATE TABLE users (
        user_id VARCHAR(50) PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        phone VARCHAR(20) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        city VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_email (email),
        INDEX idx_phone (phone)
    ) PARTITION BY HASH (user_id);
    ```

    ---

    ## Data Flow Diagrams

    ### Seat Booking Flow (Happy Path)

    ```mermaid
    sequenceDiagram
        participant User
        participant API_Gateway
        participant Booking_Service
        participant Redis_Lock
        participant Inventory_DB
        participant Payment_Service
        participant Payment_Gateway
        participant Kafka
        participant Notification_Service

        User->>API_Gateway: POST /bookings/lock (seat_ids)
        API_Gateway->>Booking_Service: Lock seats request

        Note over Booking_Service: Start distributed lock

        Booking_Service->>Redis_Lock: SETNX lock:show:seats (TTL 600s)
        Redis_Lock-->>Booking_Service: Lock acquired

        Booking_Service->>Inventory_DB: BEGIN TRANSACTION
        Booking_Service->>Inventory_DB: SELECT * FROM seats WHERE seat_id IN (...) FOR UPDATE
        Inventory_DB-->>Booking_Service: Seat rows (locked)

        alt All seats available
            Booking_Service->>Inventory_DB: UPDATE seats SET status='locked', locked_by='lock_123', locked_until=NOW()+10min
            Booking_Service->>Inventory_DB: INSERT INTO seat_locks (...)
            Booking_Service->>Inventory_DB: COMMIT
            Booking_Service->>Redis_Lock: DEL lock:show:seats
            Booking_Service-->>API_Gateway: 201 Created (lock_id, expires_at)
            API_Gateway-->>User: Lock successful
        else Some seats unavailable
            Booking_Service->>Inventory_DB: ROLLBACK
            Booking_Service->>Redis_Lock: DEL lock:show:seats
            Booking_Service-->>API_Gateway: 409 Conflict (seat unavailable)
            API_Gateway-->>User: Show alternatives
        end

        Note over User: User enters payment details (within 10 min)

        User->>API_Gateway: POST /bookings/complete (lock_id, payment_details)
        API_Gateway->>Booking_Service: Complete booking

        Booking_Service->>Inventory_DB: SELECT * FROM seat_locks WHERE lock_id=... AND status='active'

        alt Lock valid
            Booking_Service->>Payment_Service: Process payment (idempotency_key)
            Payment_Service->>Payment_Gateway: Charge card
            Payment_Gateway-->>Payment_Service: Payment success

            Payment_Service->>Inventory_DB: BEGIN TRANSACTION
            Payment_Service->>Inventory_DB: UPDATE seats SET status='booked', booking_id='bkg_123'
            Payment_Service->>Inventory_DB: INSERT INTO bookings (...)
            Payment_Service->>Inventory_DB: UPDATE seat_locks SET status='completed'
            Payment_Service->>Inventory_DB: COMMIT

            Payment_Service->>Kafka: Publish booking_confirmed event
            Payment_Service-->>Booking_Service: Booking confirmed
            Booking_Service-->>API_Gateway: 201 Created (booking_id, qr_code)
            API_Gateway-->>User: Booking successful

            Kafka->>Notification_Service: booking_confirmed event
            Notification_Service->>User: Send email/SMS with ticket
        else Lock expired
            Booking_Service-->>API_Gateway: 410 Gone (lock expired)
            API_Gateway-->>User: Lock expired, retry
        end
    ```

    **Flow Explanation:**

    1. **Acquire distributed lock** - Redis lock prevents race conditions
    2. **Check seat availability** - SELECT FOR UPDATE (pessimistic locking)
    3. **Lock seats** - Update seat status to 'locked' with expiry
    4. **Process payment** - Idempotent payment via payment gateway
    5. **Confirm booking** - Update seats to 'booked', create booking record
    6. **Send notification** - Async via Kafka (email/SMS with ticket)

    ---

    ### Lock Expiry Flow (Background Job)

    ```mermaid
    sequenceDiagram
        participant Lock_Worker
        participant Inventory_DB
        participant Redis_Lock
        participant Cache

        loop Every 30 seconds
            Lock_Worker->>Inventory_DB: SELECT * FROM seat_locks WHERE expires_at < NOW() AND status='active'
            Inventory_DB-->>Lock_Worker: Expired locks

            alt Has expired locks
                Lock_Worker->>Inventory_DB: BEGIN TRANSACTION

                loop For each expired lock
                    Lock_Worker->>Inventory_DB: UPDATE seats SET status='available', locked_by=NULL WHERE locked_by='lock_123'
                    Lock_Worker->>Inventory_DB: UPDATE seat_locks SET status='expired' WHERE lock_id='lock_123'
                end

                Lock_Worker->>Inventory_DB: COMMIT
                Lock_Worker->>Redis_Lock: DEL lock:lock_123
                Lock_Worker->>Cache: Invalidate seat availability cache

                Note over Lock_Worker: Seats now available for others
            end
        end
    ```

    **Flow Explanation:**

    1. **Periodic scan** - Every 30 seconds, find expired locks
    2. **Release seats** - Update seat status back to 'available'
    3. **Cleanup** - Remove lock records, invalidate caches
    4. **Make available** - Seats now bookable by other users

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical ticket booking subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Seat Locking Strategy** | How to prevent double booking with high concurrency? | Distributed locks + pessimistic locking + optimistic locking hybrid |
    | **Payment Integration** | How to handle payment failures and duplicates? | Idempotency, two-phase commit, retry with exponential backoff |
    | **Inventory Management** | How to maintain accurate seat counts? | ACID transactions, database constraints, event sourcing |
    | **Concurrency Handling** | How to handle 10K concurrent bookings? | Connection pooling, database partitioning, read replicas |

    ---

    === "🔒 Seat Locking Strategy"

        ## The Challenge

        **Problem:** 1,000 users try to book the same seat simultaneously. How to ensure exactly one succeeds?

        **Scenarios:**

        1. **Double booking:** Two users book same seat (unacceptable)
        2. **Lost update:** User A's booking overwrites User B's lock
        3. **Deadlock:** User A locks seat 1, User B locks seat 2, both want both

        **Requirements:**

        - **Zero double bookings** (100% guarantee)
        - **Low latency** (< 500ms for seat lock)
        - **High throughput** (10K concurrent locks)
        - **Graceful failures** (clear error messages)

        ---

        ## Locking Strategies Comparison

        | Strategy | Pros | Cons | Use Case |
        |----------|------|------|----------|
        | **Pessimistic Lock** | No conflicts, simple | Slower (holds lock), deadlock risk | **Booking (high value, low frequency)** |
        | **Optimistic Lock** | Fast reads, no deadlock | Conflicts on write, retry needed | Inventory count updates |
        | **Distributed Lock** | Coordinates across servers | Single point of failure (Redis) | **Seat lock coordination** |
        | **No Lock** | Fastest | Race conditions | Read-only queries |

        **Our approach:** **Hybrid - Distributed lock (Redis) + Pessimistic lock (DB)**

        ---

        ## Implementation: Distributed Locking (Redis)

        ```python
        import redis
        import uuid
        import time
        from contextlib import contextmanager

        class DistributedLock:
            """Redis-based distributed lock for coordinating seat booking"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.LOCK_TIMEOUT = 30  # seconds (prevent deadlocks)

            @contextmanager
            def acquire_lock(self, resource_id: str, timeout: int = 10):
                """
                Acquire distributed lock with timeout

                Args:
                    resource_id: Resource to lock (e.g., "show:shw_456:seats")
                    timeout: Max time to wait for lock (seconds)

                Yields:
                    lock_id if acquired, else raises exception
                """
                lock_key = f"lock:{resource_id}"
                lock_id = str(uuid.uuid4())
                start_time = time.time()

                # Try to acquire lock
                while time.time() - start_time < timeout:
                    # SETNX: Set if not exists (atomic)
                    acquired = self.redis.set(
                        lock_key,
                        lock_id,
                        nx=True,  # Only set if not exists
                        ex=self.LOCK_TIMEOUT  # Expire after 30s (prevent deadlocks)
                    )

                    if acquired:
                        logger.info(f"Lock acquired: {lock_key} by {lock_id}")
                        try:
                            yield lock_id
                            return
                        finally:
                            # Release lock (only if we own it)
                            self._release_lock(lock_key, lock_id)
                    else:
                        # Lock held by someone else, wait and retry
                        time.sleep(0.05)  # 50ms

                raise LockAcquisitionTimeout(f"Could not acquire lock for {resource_id}")

            def _release_lock(self, lock_key: str, lock_id: str):
                """
                Release lock (only if we own it)

                Uses Lua script for atomicity
                """
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                self.redis.eval(lua_script, 1, lock_key, lock_id)
                logger.info(f"Lock released: {lock_key}")

        class BookingService:
            """Handle seat booking with distributed locking"""

            def __init__(self, db, redis_client):
                self.db = db
                self.lock = DistributedLock(redis_client)

            def lock_seats(self, show_id: str, seat_ids: List[str], user_id: str) -> dict:
                """
                Lock seats for booking

                Args:
                    show_id: Show identifier
                    seat_ids: List of seat IDs to lock
                    user_id: User requesting lock

                Returns:
                    Lock details with expiry time

                Raises:
                    SeatUnavailableException if seats already locked/booked
                """
                # Sort seat_ids to prevent deadlocks (always lock in same order)
                seat_ids = sorted(seat_ids)

                # Acquire distributed lock for this show
                with self.lock.acquire_lock(f"show:{show_id}:seats"):
                    # Start database transaction
                    with self.db.transaction():
                        # SELECT FOR UPDATE (pessimistic lock)
                        seats = self.db.execute(
                            """
                            SELECT seat_id, status, seat_number, price
                            FROM seats
                            WHERE seat_id IN %(seat_ids)s
                            FOR UPDATE
                            """,
                            {"seat_ids": tuple(seat_ids)}
                        )

                        # Check all seats available
                        unavailable = []
                        for seat in seats:
                            if seat['status'] != 'available':
                                unavailable.append(seat['seat_id'])

                        if unavailable:
                            raise SeatUnavailableException(
                                f"Seats not available: {unavailable}"
                            )

                        # Create lock record
                        lock_id = f"lock_{uuid.uuid4().hex}"
                        expires_at = datetime.utcnow() + timedelta(minutes=10)

                        self.db.execute(
                            """
                            INSERT INTO seat_locks (lock_id, user_id, show_id, seat_ids,
                                                    total_price, expires_at, status)
                            VALUES (%(lock_id)s, %(user_id)s, %(show_id)s, %(seat_ids)s,
                                    %(total_price)s, %(expires_at)s, 'active')
                            """,
                            {
                                "lock_id": lock_id,
                                "user_id": user_id,
                                "show_id": show_id,
                                "seat_ids": seat_ids,
                                "total_price": sum(s['price'] for s in seats),
                                "expires_at": expires_at
                            }
                        )

                        # Update seat status to locked
                        self.db.execute(
                            """
                            UPDATE seats
                            SET status = 'locked',
                                locked_by = %(lock_id)s,
                                locked_until = %(expires_at)s,
                                updated_at = NOW()
                            WHERE seat_id IN %(seat_ids)s
                            """,
                            {
                                "lock_id": lock_id,
                                "expires_at": expires_at,
                                "seat_ids": tuple(seat_ids)
                            }
                        )

                        # Update show available_seats count
                        self.db.execute(
                            """
                            UPDATE shows
                            SET available_seats = available_seats - %(count)s
                            WHERE show_id = %(show_id)s
                            """,
                            {"show_id": show_id, "count": len(seat_ids)}
                        )

                        # Commit transaction
                        self.db.commit()

                        logger.info(f"Seats locked: {lock_id} for user {user_id}")

                        return {
                            "lock_id": lock_id,
                            "seats": seats,
                            "total_price": sum(s['price'] for s in seats),
                            "expires_at": expires_at.isoformat(),
                            "expires_in_seconds": 600
                        }
        ```

        ---

        ## Implementation: Optimistic Locking (Version Field)

        **Use case:** Updating seat availability counts (less critical than booking)

        ```python
        def update_seat_availability(show_id: str, count_delta: int):
            """
            Update available seats count with optimistic locking

            Args:
                show_id: Show identifier
                count_delta: Change in available seats (+/-)

            Raises:
                OptimisticLockException if version mismatch (retry)
            """
            max_retries = 3

            for attempt in range(max_retries):
                # Read current version
                row = self.db.execute(
                    "SELECT available_seats, version FROM shows WHERE show_id = %s",
                    (show_id,)
                ).fetchone()

                current_seats = row['available_seats']
                current_version = row['version']
                new_seats = current_seats + count_delta
                new_version = current_version + 1

                # Update with version check
                result = self.db.execute(
                    """
                    UPDATE shows
                    SET available_seats = %(new_seats)s,
                        version = %(new_version)s
                    WHERE show_id = %(show_id)s
                    AND version = %(current_version)s
                    """,
                    {
                        "show_id": show_id,
                        "new_seats": new_seats,
                        "new_version": new_version,
                        "current_version": current_version
                    }
                )

                if result.rowcount == 1:
                    # Success
                    logger.info(f"Updated seats for {show_id}: {new_seats}")
                    return
                else:
                    # Version mismatch, retry
                    logger.warning(f"Optimistic lock conflict, retry {attempt+1}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

            raise OptimisticLockException("Max retries exceeded")
        ```

        ---

        ## Deadlock Prevention

        **Problem:** User A wants seats [1,2], User B wants [2,3]. Both lock their first seat, then deadlock on second.

        **Solution: Always lock seats in sorted order**

        ```python
        # BAD: Can cause deadlock
        def lock_seats_bad(seat_ids):
            for seat_id in seat_ids:  # Order depends on user input
                db.execute("SELECT * FROM seats WHERE seat_id = %s FOR UPDATE", (seat_id,))

        # GOOD: Prevents deadlock
        def lock_seats_good(seat_ids):
            seat_ids = sorted(seat_ids)  # Always lock in same order
            for seat_id in seat_ids:
                db.execute("SELECT * FROM seats WHERE seat_id = %s FOR UPDATE", (seat_id,))
        ```

        ---

        ## Lock Expiry Worker

        ```python
        import schedule
        import time

        class LockExpiryWorker:
            """Background worker to release expired seat locks"""

            def __init__(self, db, cache):
                self.db = db
                self.cache = cache

            def run(self):
                """Run every 30 seconds"""
                schedule.every(30).seconds.do(self.release_expired_locks)

                while True:
                    schedule.run_pending()
                    time.sleep(1)

            def release_expired_locks(self):
                """
                Find and release expired seat locks

                Runs as periodic job to prevent seat starvation
                """
                logger.info("Running lock expiry check...")

                # Find expired locks
                expired_locks = self.db.execute(
                    """
                    SELECT lock_id, seat_ids, show_id
                    FROM seat_locks
                    WHERE expires_at < NOW()
                    AND status = 'active'
                    FOR UPDATE
                    """
                ).fetchall()

                if not expired_locks:
                    logger.info("No expired locks found")
                    return

                with self.db.transaction():
                    for lock in expired_locks:
                        # Release seats
                        self.db.execute(
                            """
                            UPDATE seats
                            SET status = 'available',
                                locked_by = NULL,
                                locked_until = NULL,
                                updated_at = NOW()
                            WHERE locked_by = %(lock_id)s
                            """,
                            {"lock_id": lock['lock_id']}
                        )

                        # Mark lock as expired
                        self.db.execute(
                            """
                            UPDATE seat_locks
                            SET status = 'expired'
                            WHERE lock_id = %(lock_id)s
                            """,
                            {"lock_id": lock['lock_id']}
                        )

                        # Update show available_seats count
                        self.db.execute(
                            """
                            UPDATE shows
                            SET available_seats = available_seats + %(count)s
                            WHERE show_id = %(show_id)s
                            """,
                            {
                                "show_id": lock['show_id'],
                                "count": len(lock['seat_ids'])
                            }
                        )

                        # Invalidate cache
                        self.cache.delete(f"seats:{lock['show_id']}")

                    self.db.commit()

                logger.info(f"Released {len(expired_locks)} expired locks")
        ```

    === "💳 Payment Integration"

        ## The Challenge

        **Problem:** Handle payment failures, network timeouts, duplicate charges, and partial failures.

        **Scenarios:**

        1. **Duplicate charge:** User clicks "Pay" twice, charged twice
        2. **Timeout:** Payment succeeds at gateway, but response times out
        3. **Partial failure:** Payment succeeds, but booking DB write fails
        4. **Retry confusion:** Retry failed payment, gateway sees duplicate

        **Requirements:**

        - **Idempotency:** Same payment request = same result (no duplicate charges)
        - **Atomicity:** Payment success ⇔ booking confirmed (two-phase commit)
        - **Retry safety:** Failed payments can be safely retried
        - **Audit trail:** Complete payment history for reconciliation

        ---

        ## Payment Flow Architecture

        **Two-phase commit pattern:**

        1. **Phase 1:** Lock seats + create pending booking
        2. **Phase 2:** Process payment + confirm booking

        ```mermaid
        stateDiagram-v2
            [*] --> SeatLocked: Lock seats
            SeatLocked --> PaymentPending: Create pending booking
            PaymentPending --> PaymentProcessing: Call payment gateway

            PaymentProcessing --> PaymentSuccess: Payment success
            PaymentProcessing --> PaymentFailed: Payment failed
            PaymentProcessing --> PaymentTimeout: Timeout

            PaymentSuccess --> BookingConfirmed: Confirm booking
            PaymentFailed --> SeatReleased: Release seats
            PaymentTimeout --> PaymentRetry: Retry payment

            PaymentRetry --> PaymentSuccess: Success after retry
            PaymentRetry --> SeatReleased: Max retries exceeded

            BookingConfirmed --> [*]
            SeatReleased --> [*]
        ```

        ---

        ## Idempotency Implementation

        **Problem:** User clicks "Pay" button twice → charged twice.

        **Solution:** Idempotency key (unique per payment attempt)

        ```python
        import hashlib
        import json

        class PaymentService:
            """Handle payment processing with idempotency"""

            def __init__(self, db, payment_gateway, cache):
                self.db = db
                self.gateway = payment_gateway
                self.cache = cache
                self.IDEMPOTENCY_TTL = 86400  # 24 hours

            def process_payment(
                self,
                lock_id: str,
                payment_method: str,
                payment_details: dict,
                idempotency_key: str
            ) -> dict:
                """
                Process payment with idempotency guarantee

                Args:
                    lock_id: Seat lock ID
                    payment_method: card, upi, wallet, etc.
                    payment_details: Payment gateway token
                    idempotency_key: Unique key for this payment attempt

                Returns:
                    Booking confirmation details

                Raises:
                    PaymentFailedException if payment fails
                    IdempotencyKeyReusedException if key already used
                """
                # Check idempotency cache
                cached_result = self.cache.get(f"payment:{idempotency_key}")
                if cached_result:
                    logger.info(f"Idempotency cache hit: {idempotency_key}")
                    return json.loads(cached_result)

                # Check database for existing transaction
                existing = self.db.execute(
                    """
                    SELECT transaction_id, status, booking_id
                    FROM payment_transactions
                    WHERE idempotency_key = %(key)s
                    """,
                    {"key": idempotency_key}
                ).fetchone()

                if existing:
                    if existing['status'] == 'success':
                        # Payment already succeeded, return existing booking
                        logger.info(f"Payment already processed: {idempotency_key}")
                        return self._get_booking_details(existing['booking_id'])
                    elif existing['status'] == 'pending':
                        # Payment in progress, wait and retry
                        raise PaymentInProgressException("Payment already being processed")

                # Validate lock
                lock = self._validate_lock(lock_id)
                if not lock:
                    raise LockExpiredException("Seat lock expired or invalid")

                # Create pending transaction record
                transaction_id = f"txn_{uuid.uuid4().hex}"
                self.db.execute(
                    """
                    INSERT INTO payment_transactions (
                        transaction_id, booking_id, payment_gateway, amount,
                        status, idempotency_key, payment_method, created_at
                    ) VALUES (
                        %(txn_id)s, NULL, %(gateway)s, %(amount)s,
                        'pending', %(idem_key)s, %(method)s, NOW()
                    )
                    """,
                    {
                        "txn_id": transaction_id,
                        "gateway": "stripe",
                        "amount": lock['total_price'],
                        "idem_key": idempotency_key,
                        "method": payment_method
                    }
                )
                self.db.commit()

                try:
                    # Call payment gateway
                    payment_result = self._charge_payment_gateway(
                        transaction_id,
                        lock['total_price'],
                        payment_details
                    )

                    if payment_result['status'] == 'success':
                        # Payment succeeded, confirm booking
                        booking = self._confirm_booking(lock, transaction_id, payment_result)

                        # Cache result for idempotency
                        self.cache.setex(
                            f"payment:{idempotency_key}",
                            self.IDEMPOTENCY_TTL,
                            json.dumps(booking)
                        )

                        return booking
                    else:
                        # Payment failed
                        self._handle_payment_failure(lock, transaction_id, payment_result)
                        raise PaymentFailedException(payment_result['error_message'])

                except Exception as e:
                    # Mark transaction as failed
                    self.db.execute(
                        """
                        UPDATE payment_transactions
                        SET status = 'failed', updated_at = NOW()
                        WHERE transaction_id = %(txn_id)s
                        """,
                        {"txn_id": transaction_id}
                    )
                    self.db.commit()
                    raise

            def _charge_payment_gateway(
                self,
                transaction_id: str,
                amount: float,
                payment_details: dict
            ) -> dict:
                """
                Call payment gateway (Stripe, Razorpay, etc.)

                Args:
                    transaction_id: Internal transaction ID
                    amount: Amount to charge
                    payment_details: Payment token/details

                Returns:
                    Payment result with status and gateway transaction ID
                """
                try:
                    # Call Stripe API (example)
                    result = self.gateway.charge(
                        amount=amount,
                        currency='INR',
                        payment_method=payment_details['token'],
                        idempotency_key=transaction_id,  # Gateway-level idempotency
                        metadata={
                            'transaction_id': transaction_id
                        }
                    )

                    return {
                        'status': 'success',
                        'gateway_txn_id': result.id,
                        'amount': result.amount,
                        'currency': result.currency
                    }

                except self.gateway.PaymentError as e:
                    return {
                        'status': 'failed',
                        'error_code': e.code,
                        'error_message': str(e)
                    }

            def _confirm_booking(
                self,
                lock: dict,
                transaction_id: str,
                payment_result: dict
            ) -> dict:
                """
                Confirm booking after successful payment

                Args:
                    lock: Seat lock details
                    transaction_id: Payment transaction ID
                    payment_result: Payment gateway response

                Returns:
                    Booking confirmation with ticket details
                """
                with self.db.transaction():
                    # Create booking
                    booking_id = f"bkg_{uuid.uuid4().hex}"
                    self.db.execute(
                        """
                        INSERT INTO bookings (
                            booking_id, user_id, show_id, seat_ids, seat_numbers,
                            total_price, status, payment_id, payment_status, booked_at
                        ) VALUES (
                            %(booking_id)s, %(user_id)s, %(show_id)s, %(seat_ids)s,
                            %(seat_numbers)s, %(total_price)s, 'confirmed',
                            %(payment_id)s, 'success', NOW()
                        )
                        """,
                        {
                            "booking_id": booking_id,
                            "user_id": lock['user_id'],
                            "show_id": lock['show_id'],
                            "seat_ids": lock['seat_ids'],
                            "seat_numbers": [s['seat_number'] for s in lock['seats']],
                            "total_price": lock['total_price'],
                            "payment_id": transaction_id
                        }
                    )

                    # Update seats to booked
                    self.db.execute(
                        """
                        UPDATE seats
                        SET status = 'booked',
                            booking_id = %(booking_id)s,
                            locked_by = NULL,
                            locked_until = NULL,
                            updated_at = NOW()
                        WHERE seat_id IN %(seat_ids)s
                        """,
                        {
                            "booking_id": booking_id,
                            "seat_ids": tuple(lock['seat_ids'])
                        }
                    )

                    # Update lock status
                    self.db.execute(
                        """
                        UPDATE seat_locks
                        SET status = 'completed'
                        WHERE lock_id = %(lock_id)s
                        """,
                        {"lock_id": lock['lock_id']}
                    )

                    # Update payment transaction
                    self.db.execute(
                        """
                        UPDATE payment_transactions
                        SET booking_id = %(booking_id)s,
                            gateway_transaction_id = %(gateway_txn_id)s,
                            status = 'success',
                            updated_at = NOW()
                        WHERE transaction_id = %(txn_id)s
                        """,
                        {
                            "booking_id": booking_id,
                            "gateway_txn_id": payment_result['gateway_txn_id'],
                            "txn_id": transaction_id
                        }
                    )

                    self.db.commit()

                # Generate QR code
                qr_code_url = self._generate_qr_code(booking_id)

                # Publish booking confirmed event (for notifications)
                self._publish_booking_event(booking_id, lock)

                return {
                    "booking_id": booking_id,
                    "status": "confirmed",
                    "qr_code": qr_code_url,
                    "ticket_url": f"https://bookmyshow.com/tickets/{booking_id}",
                    "seats": [s['seat_number'] for s in lock['seats']],
                    "total_price": lock['total_price']
                }

            def _handle_payment_failure(
                self,
                lock: dict,
                transaction_id: str,
                payment_result: dict
            ):
                """
                Handle payment failure - release seats

                Args:
                    lock: Seat lock details
                    transaction_id: Payment transaction ID
                    payment_result: Failed payment result
                """
                with self.db.transaction():
                    # Release seats
                    self.db.execute(
                        """
                        UPDATE seats
                        SET status = 'available',
                            locked_by = NULL,
                            locked_until = NULL,
                            updated_at = NOW()
                        WHERE locked_by = %(lock_id)s
                        """,
                        {"lock_id": lock['lock_id']}
                    )

                    # Update lock status
                    self.db.execute(
                        """
                        UPDATE seat_locks
                        SET status = 'expired'
                        WHERE lock_id = %(lock_id)s
                        """,
                        {"lock_id": lock['lock_id']}
                    )

                    # Update transaction
                    self.db.execute(
                        """
                        UPDATE payment_transactions
                        SET status = 'failed', updated_at = NOW()
                        WHERE transaction_id = %(txn_id)s
                        """,
                        {"txn_id": transaction_id}
                    )

                    self.db.commit()

                logger.info(f"Payment failed, seats released: {lock['lock_id']}")
        ```

        ---

        ## Retry Strategy

        **Problem:** Payment gateway timeout - did payment succeed?

        **Solution: Query payment status before retry**

        ```python
        def retry_payment_with_status_check(transaction_id: str, max_retries: int = 3):
            """
            Retry payment with status verification

            Args:
                transaction_id: Internal transaction ID
                max_retries: Maximum retry attempts

            Returns:
                Payment result
            """
            for attempt in range(max_retries):
                try:
                    # Check if payment already succeeded at gateway
                    gateway_status = self.gateway.retrieve_transaction(transaction_id)
                    if gateway_status and gateway_status.status == 'succeeded':
                        logger.info(f"Payment already succeeded: {transaction_id}")
                        return {'status': 'success', 'gateway_txn_id': gateway_status.id}

                    # Attempt payment
                    result = self._charge_payment_gateway(transaction_id, amount, details)
                    return result

                except TimeoutError:
                    logger.warning(f"Payment timeout, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                except Exception as e:
                    logger.error(f"Payment error: {e}")
                    raise

            raise PaymentTimeoutException("Payment timeout after max retries")
        ```

        ---

        ## Reconciliation Job

        **Problem:** Partial failures leave inconsistent state (payment succeeded, booking failed).

        **Solution: Nightly reconciliation job**

        ```python
        def reconcile_payments():
            """
            Nightly job to reconcile payment gateway vs database

            Finds discrepancies:
            - Payments succeeded at gateway but booking failed
            - Payments failed at gateway but marked success in DB
            """
            # Find transactions marked pending/failed but succeeded at gateway
            pending_transactions = self.db.execute(
                """
                SELECT transaction_id, gateway_transaction_id, idempotency_key
                FROM payment_transactions
                WHERE status IN ('pending', 'failed')
                AND created_at > NOW() - INTERVAL '7 days'
                """
            ).fetchall()

            for txn in pending_transactions:
                # Query gateway status
                gateway_status = self.gateway.retrieve_transaction(
                    txn['gateway_transaction_id']
                )

                if gateway_status and gateway_status.status == 'succeeded':
                    # Payment succeeded but not reflected in DB
                    logger.warning(f"Reconciliation: payment succeeded but booking failed: {txn['transaction_id']}")

                    # Complete booking (if lock still valid)
                    try:
                        self._complete_booking_from_transaction(txn['transaction_id'])
                    except Exception as e:
                        logger.error(f"Failed to complete booking: {e}")
                        # Send alert for manual investigation
                        self._send_alert(txn, gateway_status)
        ```

    === "📦 Inventory Management"

        ## The Challenge

        **Problem:** Maintain accurate seat availability across millions of seats, thousands of concurrent bookings, with zero inventory drift.

        **Requirements:**

        - **Accurate counts:** available_seats always matches reality
        - **No overselling:** Never sell more seats than exist
        - **Real-time updates:** Show latest availability to users
        - **Audit trail:** Track every inventory change

        ---

        ## Inventory Architecture

        **Multi-level inventory tracking:**

        ```
        Level 1: Show-level aggregate (fast queries)
        └── available_seats: 87 (cached)

        Level 2: Seat-level status (source of truth)
        ├── Seat A1: available
        ├── Seat A2: booked
        ├── Seat A3: locked
        └── ...

        Level 3: Event log (audit trail)
        ├── 14:30:00 - Seat A1 locked by user_123
        ├── 14:35:00 - Seat A1 booked by user_123
        └── ...
        ```

        ---

        ## Database Constraints

        **Problem:** Application bug causes negative available_seats.

        **Solution: Database-level constraints**

        ```sql
        -- Constraint: available_seats cannot be negative
        ALTER TABLE shows
        ADD CONSTRAINT chk_available_seats
        CHECK (available_seats >= 0 AND available_seats <= total_seats);

        -- Trigger: Automatically update available_seats on seat status change
        CREATE OR REPLACE FUNCTION update_show_available_seats()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'UPDATE' THEN
                -- Seat status changed
                IF OLD.status = 'available' AND NEW.status IN ('locked', 'booked') THEN
                    -- Seat became unavailable
                    UPDATE shows
                    SET available_seats = available_seats - 1
                    WHERE show_id = NEW.show_id;
                ELSIF OLD.status IN ('locked', 'booked') AND NEW.status = 'available' THEN
                    -- Seat became available
                    UPDATE shows
                    SET available_seats = available_seats + 1
                    WHERE show_id = NEW.show_id;
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_update_available_seats
        AFTER UPDATE OF status ON seats
        FOR EACH ROW
        EXECUTE FUNCTION update_show_available_seats();
        ```

        ---

        ## Event Sourcing (Audit Trail)

        **Problem:** How to audit inventory changes? Who locked this seat? When?

        **Solution: Event log for all inventory changes**

        ```sql
        -- Inventory events (immutable log)
        CREATE TABLE inventory_events (
            event_id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL, -- seat_locked, seat_booked, seat_released, etc.
            show_id VARCHAR(50) NOT NULL,
            seat_id VARCHAR(50) NOT NULL,
            user_id VARCHAR(50),
            lock_id VARCHAR(50),
            booking_id VARCHAR(50),
            previous_status VARCHAR(50),
            new_status VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB,
            INDEX idx_seat_events (seat_id, timestamp),
            INDEX idx_show_events (show_id, timestamp)
        );

        -- Trigger: Log all seat status changes
        CREATE OR REPLACE FUNCTION log_inventory_event()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
                INSERT INTO inventory_events (
                    event_type, show_id, seat_id, previous_status, new_status,
                    lock_id, booking_id, timestamp
                ) VALUES (
                    CASE
                        WHEN NEW.status = 'locked' THEN 'seat_locked'
                        WHEN NEW.status = 'booked' THEN 'seat_booked'
                        WHEN NEW.status = 'available' THEN 'seat_released'
                    END,
                    NEW.show_id,
                    NEW.seat_id,
                    OLD.status,
                    NEW.status,
                    NEW.locked_by,
                    NEW.booking_id,
                    NOW()
                );
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_log_inventory_event
        AFTER UPDATE ON seats
        FOR EACH ROW
        EXECUTE FUNCTION log_inventory_event();
        ```

        ---

        ## Inventory Reconciliation

        **Problem:** Cache/DB drift - show.available_seats doesn't match actual seat count.

        **Solution: Periodic reconciliation job**

        ```python
        class InventoryReconciliationJob:
            """Reconcile show-level counts with seat-level status"""

            def __init__(self, db):
                self.db = db

            def run_reconciliation(self):
                """
                Compare show.available_seats with actual seat counts

                Runs: Daily at 3 AM (low traffic)
                """
                logger.info("Starting inventory reconciliation...")

                # Find shows with mismatched counts
                mismatches = self.db.execute(
                    """
                    WITH actual_counts AS (
                        SELECT
                            show_id,
                            COUNT(*) FILTER (WHERE status = 'available') as actual_available,
                            COUNT(*) as total
                        FROM seats
                        GROUP BY show_id
                    )
                    SELECT
                        s.show_id,
                        s.available_seats as reported_available,
                        ac.actual_available,
                        s.total_seats,
                        ac.total
                    FROM shows s
                    JOIN actual_counts ac ON s.show_id = ac.show_id
                    WHERE s.available_seats != ac.actual_available
                       OR s.total_seats != ac.total
                    """
                ).fetchall()

                if not mismatches:
                    logger.info("No inventory mismatches found")
                    return

                # Fix mismatches
                for mismatch in mismatches:
                    logger.warning(
                        f"Inventory mismatch for show {mismatch['show_id']}: "
                        f"reported={mismatch['reported_available']}, "
                        f"actual={mismatch['actual_available']}"
                    )

                    # Update show counts
                    self.db.execute(
                        """
                        UPDATE shows
                        SET available_seats = %(actual)s
                        WHERE show_id = %(show_id)s
                        """,
                        {
                            "show_id": mismatch['show_id'],
                            "actual": mismatch['actual_available']
                        }
                    )

                    # Log reconciliation event
                    self.db.execute(
                        """
                        INSERT INTO inventory_events (
                            event_type, show_id, metadata, timestamp
                        ) VALUES (
                            'reconciliation_correction',
                            %(show_id)s,
                            %(metadata)s,
                            NOW()
                        )
                        """,
                        {
                            "show_id": mismatch['show_id'],
                            "metadata": json.dumps({
                                "reported": mismatch['reported_available'],
                                "actual": mismatch['actual_available'],
                                "difference": mismatch['actual_available'] - mismatch['reported_available']
                            })
                        }
                    )

                    # Invalidate cache
                    self.cache.delete(f"seats:{mismatch['show_id']}")

                self.db.commit()
                logger.info(f"Fixed {len(mismatches)} inventory mismatches")
        ```

        ---

        ## Caching Strategy

        **Problem:** Querying seat availability for every request overloads DB.

        **Solution: Multi-tier caching**

        ```python
        class SeatAvailabilityCache:
            """Cache seat availability with smart invalidation"""

            def __init__(self, redis_client, db):
                self.cache = redis_client
                self.db = db
                self.CACHE_TTL = 30  # seconds

            def get_available_seats(self, show_id: str) -> List[dict]:
                """
                Get seat availability (cached)

                Cache invalidated on:
                - Seat lock/booking
                - Lock expiry
                - Booking cancellation
                """
                cache_key = f"seats:{show_id}"

                # Check cache
                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached)

                # Cache miss - query DB
                seats = self.db.execute(
                    """
                    SELECT seat_id, seat_number, row_name, seat_type,
                           price, status, locked_until
                    FROM seats
                    WHERE show_id = %(show_id)s
                    ORDER BY row_name, seat_number
                    """,
                    {"show_id": show_id}
                ).fetchall()

                # Cache for 30 seconds
                self.cache.setex(cache_key, self.CACHE_TTL, json.dumps(seats))

                return seats

            def invalidate_show_cache(self, show_id: str):
                """Invalidate cache on seat status change"""
                self.cache.delete(f"seats:{show_id}")
                logger.info(f"Invalidated cache for show {show_id}")

            def get_show_summary(self, show_id: str) -> dict:
                """
                Get show-level summary (heavily cached)

                Cache TTL: 60 seconds (less critical than seat-level)
                """
                cache_key = f"show_summary:{show_id}"

                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached)

                summary = self.db.execute(
                    """
                    SELECT show_id, available_seats, total_seats
                    FROM shows
                    WHERE show_id = %(show_id)s
                    """,
                    {"show_id": show_id}
                ).fetchone()

                self.cache.setex(cache_key, 60, json.dumps(summary))
                return summary
        ```

    === "⚡ Concurrency Handling"

        ## The Challenge

        **Problem:** 10,000 concurrent users booking tickets for same movie (Friday release). Handle peak load without failures.

        **Challenges:**

        - **Database connection exhaustion** (1000 concurrent bookings × 5s each = 5000 connections)
        - **Lock contention** (everyone wants front-row seats)
        - **Payment gateway rate limits** (100 requests/second)
        - **Hot partitions** (everyone queries same show)

        ---

        ## Database Connection Pooling

        **Problem:** Each booking holds DB connection for 5s (lock → payment → confirm). 10K bookings = need 50K connections!

        **Solution: Connection pooling + async I/O**

        ```python
        import asyncpg
        import asyncio

        class DatabasePool:
            """Async database connection pool"""

            def __init__(self, dsn: str):
                self.dsn = dsn
                self.pool = None

            async def initialize(self):
                """
                Create connection pool

                Pool size: 100 connections (shared across all requests)
                """
                self.pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=20,    # Keep 20 connections open
                    max_size=100,   # Max 100 concurrent connections
                    max_queries=50000,  # Max queries per connection
                    max_inactive_connection_lifetime=300,  # Close idle connections
                    command_timeout=10  # Query timeout
                )

            async def execute(self, query: str, *args):
                """Execute query from pool"""
                async with self.pool.acquire() as conn:
                    return await conn.fetch(query, *args)

        class AsyncBookingService:
            """Async booking service for high concurrency"""

            def __init__(self, db_pool, redis):
                self.db = db_pool
                self.redis = redis

            async def lock_seats(self, show_id: str, seat_ids: List[str], user_id: str):
                """
                Lock seats asynchronously

                Allows handling 10K concurrent requests with 100 DB connections
                """
                seat_ids = sorted(seat_ids)

                # Acquire distributed lock (Redis)
                lock_acquired = await self.redis.set(
                    f"lock:show:{show_id}:seats",
                    user_id,
                    nx=True,
                    ex=30
                )

                if not lock_acquired:
                    raise LockContentionException("Another booking in progress")

                try:
                    # Start transaction
                    async with self.db.pool.acquire() as conn:
                        async with conn.transaction():
                            # SELECT FOR UPDATE
                            seats = await conn.fetch(
                                """
                                SELECT seat_id, status, price
                                FROM seats
                                WHERE seat_id = ANY($1::text[])
                                FOR UPDATE
                                """,
                                seat_ids
                            )

                            # Check availability
                            unavailable = [
                                s['seat_id'] for s in seats
                                if s['status'] != 'available'
                            ]

                            if unavailable:
                                raise SeatUnavailableException(unavailable)

                            # Create lock
                            lock_id = f"lock_{uuid.uuid4().hex}"
                            expires_at = datetime.utcnow() + timedelta(minutes=10)

                            await conn.execute(
                                """
                                INSERT INTO seat_locks (lock_id, user_id, show_id,
                                                        seat_ids, total_price, expires_at)
                                VALUES ($1, $2, $3, $4, $5, $6)
                                """,
                                lock_id, user_id, show_id, seat_ids,
                                sum(s['price'] for s in seats), expires_at
                            )

                            # Update seats
                            await conn.execute(
                                """
                                UPDATE seats
                                SET status = 'locked',
                                    locked_by = $1,
                                    locked_until = $2
                                WHERE seat_id = ANY($3::text[])
                                """,
                                lock_id, expires_at, seat_ids
                            )

                            return {
                                "lock_id": lock_id,
                                "expires_at": expires_at.isoformat()
                            }

                finally:
                    # Release distributed lock
                    await self.redis.delete(f"lock:show:{show_id}:seats")
        ```

        ---

        ## Rate Limiting

        **Problem:** 10K concurrent requests to payment gateway = rate limit errors.

        **Solution: Token bucket rate limiter**

        ```python
        import time
        from collections import deque

        class TokenBucketRateLimiter:
            """Token bucket rate limiter for payment gateway"""

            def __init__(self, rate: int, capacity: int):
                """
                Args:
                    rate: Tokens added per second (e.g., 100)
                    capacity: Max tokens in bucket (e.g., 200)
                """
                self.rate = rate
                self.capacity = capacity
                self.tokens = capacity
                self.last_refill = time.time()
                self.lock = asyncio.Lock()

            async def acquire(self, tokens: int = 1) -> bool:
                """
                Try to acquire tokens

                Args:
                    tokens: Number of tokens to acquire

                Returns:
                    True if acquired, False if rate limited
                """
                async with self.lock:
                    # Refill tokens
                    now = time.time()
                    elapsed = now - self.last_refill
                    refill = elapsed * self.rate
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last_refill = now

                    # Try to acquire
                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        return True
                    else:
                        return False

        class RateLimitedPaymentService:
            """Payment service with rate limiting"""

            def __init__(self, gateway, rate_limit):
                self.gateway = gateway
                self.limiter = TokenBucketRateLimiter(
                    rate=100,      # 100 payments/second
                    capacity=200   # Burst capacity
                )

            async def process_payment(self, amount: float, details: dict):
                """
                Process payment with rate limiting

                If rate limited, queue for later processing
                """
                # Try to acquire token
                max_wait = 30  # seconds
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    if await self.limiter.acquire():
                        # Rate limit available, process payment
                        return await self.gateway.charge(amount, details)
                    else:
                        # Rate limited, wait and retry
                        await asyncio.sleep(0.1)

                raise RateLimitException("Payment gateway rate limit exceeded")
        ```

        ---

        ## Database Sharding

        **Problem:** Hot partition - everyone queries shows in "Bangalore" city.

        **Solution: Shard by show_id (not city)**

        ```python
        class ShardedDatabase:
            """Shard database by show_id"""

            def __init__(self, shard_count: int = 16):
                self.shard_count = shard_count
                self.shards = [
                    DatabasePool(f"postgresql://host{i}/booking_shard_{i}")
                    for i in range(shard_count)
                ]

            def get_shard(self, show_id: str) -> DatabasePool:
                """
                Determine shard for show_id

                Uses consistent hashing for even distribution
                """
                shard_id = hash(show_id) % self.shard_count
                return self.shards[shard_id]

            async def get_seats(self, show_id: str):
                """Query seats from correct shard"""
                shard = self.get_shard(show_id)
                return await shard.execute(
                    "SELECT * FROM seats WHERE show_id = $1",
                    show_id
                )

            async def lock_seats(self, show_id: str, seat_ids: List[str], user_id: str):
                """Lock seats in correct shard"""
                shard = self.get_shard(show_id)
                # ... locking logic
        ```

        **Benefits:**

        - **Distributes load** across 16 shards
        - **No hot partitions** (show_id evenly distributed)
        - **Horizontal scaling** (add more shards as needed)

        ---

        ## Read Replicas

        **Problem:** Search queries overload primary database.

        **Solution: Read replicas for read-heavy queries**

        ```python
        class DatabaseWithReplicas:
            """Primary DB + read replicas"""

            def __init__(self, primary_dsn: str, replica_dsns: List[str]):
                self.primary = DatabasePool(primary_dsn)
                self.replicas = [DatabasePool(dsn) for dsn in replica_dsns]
                self.replica_index = 0

            def get_read_replica(self) -> DatabasePool:
                """Round-robin replica selection"""
                replica = self.replicas[self.replica_index]
                self.replica_index = (self.replica_index + 1) % len(self.replicas)
                return replica

            async def search_events(self, city: str, date: str):
                """Search events (read from replica)"""
                replica = self.get_read_replica()
                return await replica.execute(
                    """
                    SELECT e.*, s.theater_id, s.start_time
                    FROM events e
                    JOIN shows s ON e.event_id = s.event_id
                    JOIN theaters t ON s.theater_id = t.theater_id
                    WHERE t.city = $1 AND DATE(s.start_time) = $2
                    """,
                    city, date
                )

            async def lock_seats(self, show_id: str, seat_ids: List[str]):
                """Lock seats (write to primary)"""
                return await self.primary.execute(
                    "UPDATE seats SET status = 'locked' WHERE seat_id = ANY($1)",
                    seat_ids
                )
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling BookMyShow from 1M to 50M bookings/month.

    **Scaling challenges at 50M bookings/month:**

    - **Peak concurrency:** 10K concurrent bookings (Friday evening releases)
    - **Database load:** 193 booking QPS + 640 lock QPS = 833 write QPS
    - **Lock contention:** Thousands of users competing for same seats
    - **Payment throughput:** 193 payment gateway calls/second

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **PostgreSQL writes** | ✅ Yes | Shard by show_id (16 shards), connection pooling (100 per shard) |
    | **Redis locks** | 🟡 Approaching | Redis cluster (3 master nodes), lock TTL tuning |
    | **Payment gateway** | ✅ Yes | Rate limiting (100/s), async processing, retry queue |
    | **Seat availability cache** | ✅ Yes | Multi-tier caching (Redis L1, in-memory L2), 30s TTL |
    | **Search queries** | 🟡 Moderate | Elasticsearch (12 shards), read replicas (5 replicas) |

    ---

    ## Scaling Strategies

    ### 1. Database Sharding

    **Problem:** Single PostgreSQL instance maxes out at ~1K write QPS.

    **Solution: Shard by show_id**

    ```
    16 shards (show_id % 16)
    ├── Shard 0: shows 0, 16, 32, ... (833/16 = 52 write QPS)
    ├── Shard 1: shows 1, 17, 33, ...
    └── ...

    Each shard:
    - PostgreSQL 13 (m5.2xlarge)
    - 100 connection pool
    - 5 read replicas
    ```

    ### 2. Redis Clustering

    **Problem:** Single Redis instance handles 640 lock requests/second.

    **Solution: Redis cluster**

    ```
    Redis Cluster (3 master nodes)
    ├── Master 1: locks for shows 0-33% (213 locks/s)
    ├── Master 2: locks for shows 34-66% (213 locks/s)
    └── Master 3: locks for shows 67-100% (214 locks/s)

    Each master:
    - Redis 7 (r5.xlarge)
    - 2 replicas (for availability)
    ```

    ### 3. Payment Gateway Throttling

    **Problem:** Payment gateway rate limit (100 requests/second).

    **Solution: Queue-based processing**

    ```python
    async def queue_payment(booking_request):
        """Queue payment for async processing"""
        await kafka.produce(
            topic='payment_queue',
            key=booking_request['lock_id'],
            value=booking_request
        )
        return {"status": "payment_queued"}

    async def payment_worker():
        """Process payments from queue with rate limiting"""
        rate_limiter = TokenBucketRateLimiter(rate=100, capacity=200)

        async for message in kafka.consume('payment_queue'):
            # Wait for rate limit token
            while not await rate_limiter.acquire():
                await asyncio.sleep(0.1)

            # Process payment
            await process_payment(message.value)
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 50M bookings/month:**

    | Component | Cost |
    |-----------|------|
    | **API servers** | $4,320 (30 × m5.large) |
    | **PostgreSQL (16 shards)** | $17,280 (16 × m5.2xlarge) |
    | **PostgreSQL read replicas** | $43,200 (80 × m5.large, 5 per shard) |
    | **Redis cluster** | $4,320 (3 × r5.xlarge × 3 replicas = 9 nodes) |
    | **Elasticsearch** | $12,960 (12 × m5.xlarge) |
    | **Kafka cluster** | $2,160 (6 × m5.large) |
    | **S3 storage** | $500 (10 TB for QR codes, tickets) |
    | **CDN** | $1,000 (image assets) |
    | **Payment gateway fees** | $250,000 (50M × $0.50 fee/booking) |
    | **Total** | **$335,740/month** |

    **Revenue (2% commission):**
    - 50M bookings × ₹300 avg × 2% = ₹300M/month ($3.6M/month)
    - Profit margin: ~91% (after infrastructure costs)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Seat Lock Latency (P95)** | < 500ms | > 1s |
    | **Double Booking Count** | 0 | > 0 (critical alert) |
    | **Lock Expiry Success Rate** | > 99% | < 95% |
    | **Payment Success Rate** | > 95% | < 90% |
    | **Database Connection Pool** | < 80% utilized | > 90% |
    | **Redis Lock Contention** | < 5% retry rate | > 20% |
    | **Inventory Drift** | 0 (reconciliation) | > 10 shows/day |

    ---

    ## Disaster Recovery

    **RTO (Recovery Time Objective):** 15 minutes

    **RPO (Recovery Point Objective):** 5 minutes

    **Strategies:**

    1. **Database backups:** Hourly incremental, daily full backup
    2. **Point-in-time recovery:** 7-day retention
    3. **Multi-region replication:** Primary (us-east-1), DR (us-west-2)
    4. **Automated failover:** Health checks every 30s, auto-failover on failure

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Distributed locks (Redis) + pessimistic locks (DB):** Prevent race conditions and double booking
    2. **Strong consistency (PostgreSQL):** ACID transactions for seat inventory
    3. **Idempotency:** Prevent duplicate payment charges
    4. **Lock expiry worker:** Auto-release abandoned seat locks
    5. **Event sourcing:** Complete audit trail for inventory changes
    6. **Database sharding:** Scale write throughput by show_id
    7. **Async payment processing:** Handle payment gateway rate limits

    ---

    ## Interview Tips

    ✅ **Emphasize consistency** - No double booking tolerance, use ACID

    ✅ **Explain locking strategies** - Distributed + pessimistic + optimistic

    ✅ **Discuss race conditions** - How to prevent with locks and transactions

    ✅ **Payment idempotency** - Critical for financial transactions

    ✅ **Inventory reconciliation** - Database constraints + periodic jobs

    ✅ **Concurrency handling** - Connection pooling, sharding, rate limiting

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent double booking?"** | Distributed lock (Redis) + SELECT FOR UPDATE + ACID transactions |
    | **"What if lock expires during payment?"** | Validate lock before payment, extend if needed, fail gracefully |
    | **"How to handle payment failures?"** | Idempotency key, retry with status check, release seats on failure |
    | **"Pessimistic vs optimistic locking?"** | Pessimistic for booking (high value), optimistic for inventory counts |
    | **"How to scale to 100K concurrent bookings?"** | More shards (32+), Redis cluster, async payment queue, regional deployment |
    | **"What if Redis fails?"** | Fall back to DB-only locking (slower but safe), Redis persistence (AOF) |
    | **"How to handle hot shows?"** | Queue-based booking, waitlist, increase lock timeout pressure |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** BookMyShow, Ticketmaster, Eventbrite, StubHub, Fandango, Paytm

---

*Master this problem and you'll be ready for: Airline booking, hotel reservation, restaurant reservation, ride-sharing seat selection*
