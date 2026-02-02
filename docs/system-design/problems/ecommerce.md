# Design E-Commerce Platform (Amazon)

A large-scale e-commerce platform where users can browse products, add items to cart, make purchases, track orders, and leave reviews. The system handles millions of concurrent users, billions of products, and processes millions of transactions daily.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 300M customers, 100M products, 2M orders/day, 10M concurrent sessions |
| **Key Challenges** | Inventory consistency, cart management, payment processing, order fulfillment, flash sales |
| **Core Concepts** | ACID transactions, distributed locking, inventory reservation, payment idempotency, eventual consistency |
| **Companies** | Amazon, eBay, Shopify, Walmart, Alibaba, Target, Etsy |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Product Catalog** | Browse, search, filter products by category/price | P0 (Must have) |
    | **Shopping Cart** | Add/remove items, save for later | P0 (Must have) |
    | **Checkout** | Place order, payment processing | P0 (Must have) |
    | **Order Management** | View order history, track shipments | P0 (Must have) |
    | **Inventory Management** | Track stock levels, prevent overselling | P0 (Must have) |
    | **Payment Processing** | Credit card, digital wallets, refunds | P0 (Must have) |
    | **Product Reviews** | Rate and review products | P1 (Should have) |
    | **Recommendations** | Personalized product suggestions | P1 (Should have) |
    | **Wishlist** | Save products for future purchase | P2 (Nice to have) |
    | **Search** | Full-text search, autocomplete, filters | P0 (Must have) |

    **Explicitly Out of Scope** (mention in interview):

    - Seller onboarding and management
    - Marketing and ads platform
    - Customer support ticketing
    - Warehouse management system
    - Logistics and delivery routing
    - Fraud detection ML models (basic checks only)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | E-commerce downtime = direct revenue loss |
    | **Latency (Browse)** | < 200ms p95 | Fast browsing critical for conversion |
    | **Latency (Checkout)** | < 2s p95 | Quick checkout reduces cart abandonment |
    | **Consistency (Inventory)** | Strong consistency | Prevent overselling (negative user experience) |
    | **Consistency (Orders)** | Strong consistency | Financial transactions must be ACID |
    | **Consistency (Catalog)** | Eventual consistency | Brief delays acceptable for product updates |
    | **Scalability** | Handle flash sales (10x normal traffic) | Black Friday, Prime Day events |
    | **Data Durability** | 99.999999999% | Order and payment data cannot be lost |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Monthly Active Users (MAU): 500M
    Daily Active Users (DAU): 300M
    Concurrent users (peak): 10M

    Product browsing:
    - Page views per DAU: ~20 pages/day
    - Daily page views: 300M × 20 = 6B page views/day
    - Browse QPS: 6B / 86,400 = ~69,400 req/sec
    - Peak QPS: 3x average = ~208,000 req/sec (flash sales)

    Search queries:
    - Searches per DAU: ~5 searches/day
    - Daily searches: 300M × 5 = 1.5B searches/day
    - Search QPS: 1.5B / 86,400 = ~17,400 req/sec

    Cart operations:
    - Cart updates per DAU: ~3 updates/day
    - Daily cart ops: 300M × 3 = 900M ops/day
    - Cart QPS: 900M / 86,400 = ~10,400 req/sec

    Orders:
    - Orders per day: 2M orders/day
    - Order QPS: 2M / 86,400 = ~23 orders/sec
    - Peak QPS: 10x average = ~230 orders/sec (flash sales)
    - Conversion rate: 2M / 300M = 0.67%

    Inventory checks:
    - Every cart add + checkout = 900M + 2M = 902M/day
    - Inventory QPS: 902M / 86,400 = ~10,440 req/sec

    Total Read QPS: ~97K (browse + search + inventory checks)
    Total Write QPS: ~11K (cart + orders + inventory updates)
    Read/Write ratio: 9:1 (read-heavy)
    ```

    ### Storage Estimates

    ```
    Product catalog:
    - Products: 100M products
    - Product data: 10 KB (title, description, specs, images URLs)
    - Total: 100M × 10 KB = 1 TB

    Product images/videos:
    - 10 images per product (avg)
    - Image size: 500 KB (compressed)
    - Total: 100M × 10 × 500 KB = 500 TB

    User data:
    - Users: 500M users
    - User profile: 2 KB (name, email, address, payment methods)
    - Total: 500M × 2 KB = 1 TB

    Shopping carts:
    - Active carts: 10M (concurrent users)
    - Cart size: 5 KB (10 items × 500 bytes)
    - Total: 10M × 5 KB = 50 GB

    Order history:
    - Orders per year: 2M × 365 = 730M orders/year
    - Order data: 5 KB (items, amounts, address, status)
    - 10 years: 730M × 10 × 5 KB = 36.5 TB

    Reviews:
    - 30% of orders result in review: 730M × 0.3 = 219M reviews/year
    - Review size: 2 KB (rating, text, images)
    - 10 years: 219M × 10 × 2 KB = 4.4 TB

    Inventory records:
    - Products: 100M products
    - Inventory data: 1 KB (quantity, reserved, location)
    - Total: 100M × 1 KB = 100 GB

    Total: 1 TB (catalog) + 500 TB (images) + 1 TB (users) +
           36.5 TB (orders) + 4.4 TB (reviews) + 100 GB (inventory)
           ≈ 543 TB
    ```

    ### Bandwidth Estimates

    ```
    Product browsing ingress:
    - Minimal (mostly reads)

    Product browsing egress:
    - 69,400 req/sec × 50 KB (avg page) = 3.47 GB/sec ≈ 28 Gbps
    - Images: 69,400 × 5 images × 500 KB = 173 GB/sec ≈ 1,384 Gbps

    Order creation ingress:
    - 23 orders/sec × 5 KB = 115 KB/sec ≈ 1 Mbps

    Search egress:
    - 17,400 req/sec × 20 KB (results) = 348 MB/sec ≈ 2.8 Gbps

    Total ingress: ~10 Mbps (mostly writes)
    Total egress: ~1,415 Gbps (CDN critical for images)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot products (80/20 rule):
    - 20% of products = 20M products
    - 20M × 10 KB = 200 GB

    Search index (in-memory):
    - Recent searches: 100M queries × 100 bytes = 10 GB

    Inventory cache (hot products):
    - 20M products × 1 KB = 20 GB

    User sessions:
    - 10M concurrent users × 10 KB = 100 GB

    Shopping carts:
    - 10M active carts × 5 KB = 50 GB

    Price cache:
    - 100M products × 200 bytes = 20 GB

    Total cache: 200 GB + 10 GB + 20 GB + 100 GB + 50 GB + 20 GB ≈ 400 GB
    ```

    ---

    ## Key Assumptions

    1. Average order value: $50
    2. Items per order: 3 items (avg)
    3. Conversion rate: 0.67% (2M orders from 300M DAU)
    4. Peak traffic during flash sales: 10x normal
    5. Inventory updates must be strongly consistent
    6. Payment processing requires ACID guarantees
    7. Product catalog can be eventually consistent

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Strong consistency for critical paths:** Inventory, orders, payments
    2. **Eventual consistency for catalog:** Product info, reviews
    3. **Idempotent operations:** Payments, order creation
    4. **Horizontal scalability:** Stateless services, database sharding
    5. **Isolation of concerns:** Separate read/write paths

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets, images]
            API_Gateway[API Gateway<br/>Rate limiting, auth]
        end

        subgraph "API Layer"
            Catalog_API[Catalog Service<br/>Browse products]
            Search_API[Search Service<br/>Product search]
            Cart_API[Cart Service<br/>Shopping cart]
            Order_API[Order Service<br/>Order management]
            Payment_API[Payment Service<br/>Payment processing]
            Inventory_API[Inventory Service<br/>Stock management]
            Review_API[Review Service<br/>Reviews & ratings]
            User_API[User Service<br/>User profiles]
        end

        subgraph "Data Processing"
            Inventory_Worker[Inventory Worker<br/>Stock updates]
            Order_Worker[Order Worker<br/>Order processing]
            Recommendation_Engine[Recommendation Engine<br/>ML pipeline]
            Analytics[Analytics Pipeline<br/>Sales reporting]
        end

        subgraph "Caching"
            Redis_Product[Redis<br/>Product cache]
            Redis_Cart[Redis<br/>Cart cache]
            Redis_Session[Redis<br/>Session store]
            Redis_Inventory[Redis<br/>Inventory cache]
        end

        subgraph "Storage"
            Product_DB[(Product DB<br/>PostgreSQL<br/>Read replicas)]
            Order_DB[(Order DB<br/>PostgreSQL<br/>Sharded)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Inventory_DB[(Inventory DB<br/>PostgreSQL<br/>Strong consistency)]
            Search_DB[(Elasticsearch<br/>Product search)]
            S3[Object Storage<br/>S3<br/>Images, videos)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "External Services"
            Payment_Gateway[Payment Gateway<br/>Stripe/PayPal]
            Shipping_API[Shipping API<br/>UPS/FedEx]
            Email_Service[Email Service<br/>SendGrid]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> API_Gateway
        Web --> API_Gateway

        CDN --> S3

        API_Gateway --> Catalog_API
        API_Gateway --> Search_API
        API_Gateway --> Cart_API
        API_Gateway --> Order_API
        API_Gateway --> Payment_API
        API_Gateway --> Inventory_API
        API_Gateway --> Review_API
        API_Gateway --> User_API

        Catalog_API --> Redis_Product
        Catalog_API --> Product_DB
        Catalog_API --> S3

        Search_API --> Search_DB
        Search_API --> Redis_Product

        Cart_API --> Redis_Cart
        Cart_API --> Inventory_API

        Order_API --> Order_DB
        Order_API --> Kafka
        Order_API --> Inventory_API
        Order_API --> Payment_API

        Payment_API --> Order_DB
        Payment_API --> Payment_Gateway

        Inventory_API --> Redis_Inventory
        Inventory_API --> Inventory_DB

        Review_API --> Product_DB

        User_API --> User_DB
        User_API --> Redis_Session

        Kafka --> Inventory_Worker
        Kafka --> Order_Worker
        Kafka --> Recommendation_Engine
        Kafka --> Analytics

        Order_Worker --> Email_Service
        Order_Worker --> Shipping_API

        Inventory_Worker --> Inventory_DB

        style CDN fill:#e8f5e9
        style API_Gateway fill:#e1f5ff
        style Redis_Product fill:#fff4e1
        style Redis_Cart fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Redis_Inventory fill:#fff4e1
        style Product_DB fill:#ffe1e1
        style Order_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Inventory_DB fill:#ffe1e1
        style Search_DB fill:#e8eaf6
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PostgreSQL (Orders)** | ACID transactions for orders/payments, strong consistency | NoSQL (can't guarantee ACID), MySQL (PostgreSQL better for JSON) |
    | **Redis (Cart/Session)** | In-memory speed for hot data, TTL for cart expiry | Database (too slow for cart ops), Memcached (limited data structures) |
    | **Elasticsearch** | Fast product search (<100ms), faceted filtering, autocomplete | Database full-text (too slow), Algolia (cost at scale) |
    | **Kafka** | Reliable event streaming, order processing pipeline | RabbitMQ (lower throughput), direct calls (no retry/replay) |
    | **CDN** | Image delivery (1,384 Gbps bandwidth), reduce origin load | Origin servers (can't handle bandwidth) |
    | **API Gateway** | Rate limiting (prevent abuse), authentication, routing | Individual service auth (inconsistent, redundant) |

    **Key Trade-off:** We chose **strong consistency for inventory** over performance. Better to show "out of stock" than oversell and disappoint customers.

    ---

    ## API Design

    ### 1. Get Product Details

    **Request:**
    ```http
    GET /api/v1/products/12345
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "product_id": "12345",
      "name": "Wireless Bluetooth Headphones",
      "description": "Premium noise-cancelling headphones...",
      "brand": "TechBrand",
      "category": "Electronics > Audio",
      "price": {
        "amount": 99.99,
        "currency": "USD",
        "discount": {
          "percentage": 15,
          "final_price": 84.99
        }
      },
      "inventory": {
        "in_stock": true,
        "quantity": 42,
        "warehouse_location": "US-EAST-1"
      },
      "images": [
        "https://cdn.shop.com/products/12345/img1.jpg",
        "https://cdn.shop.com/products/12345/img2.jpg"
      ],
      "specifications": {
        "battery_life": "30 hours",
        "weight": "250g",
        "connectivity": "Bluetooth 5.0"
      },
      "rating": {
        "average": 4.5,
        "count": 1823
      },
      "shipping": {
        "estimated_delivery": "2-3 business days",
        "free_shipping": true
      }
    }
    ```

    **Design Notes:**

    - Cache product details for 5 minutes (eventual consistency OK)
    - Inventory fetched from separate service (strong consistency)
    - Include shipping info for conversion optimization

    ---

    ### 2. Add to Cart

    **Request:**
    ```http
    POST /api/v1/cart/items
    Authorization: Bearer <token>
    Content-Type: application/json

    {
      "product_id": "12345",
      "quantity": 2,
      "selected_options": {
        "color": "Black",
        "size": "Medium"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "cart_id": "cart_abc123",
      "items": [
        {
          "cart_item_id": "item_xyz789",
          "product_id": "12345",
          "name": "Wireless Bluetooth Headphones",
          "quantity": 2,
          "unit_price": 84.99,
          "subtotal": 169.98,
          "image": "https://cdn.shop.com/products/12345/img1.jpg"
        }
      ],
      "summary": {
        "items_count": 2,
        "subtotal": 169.98,
        "tax": 15.30,
        "shipping": 0.00,
        "total": 185.28
      },
      "expires_at": "2026-02-09T10:30:00Z"
    }
    ```

    **Design Notes:**

    - Check inventory availability before adding
    - Cart stored in Redis with 30-day TTL
    - Return full cart summary for UI update
    - Idempotent: same product + quantity = update not duplicate

    ---

    ### 3. Checkout / Create Order

    **Request:**
    ```http
    POST /api/v1/orders
    Authorization: Bearer <token>
    Content-Type: application/json
    Idempotency-Key: order_20260202_abc123

    {
      "cart_id": "cart_abc123",
      "shipping_address": {
        "name": "John Doe",
        "street": "123 Main St",
        "city": "Seattle",
        "state": "WA",
        "zip": "98101",
        "country": "US"
      },
      "payment_method": {
        "type": "credit_card",
        "card_id": "card_xyz789"
      },
      "shipping_method": "standard"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "order_id": "order_456789",
      "status": "processing",
      "created_at": "2026-02-02T10:30:00Z",
      "items": [
        {
          "product_id": "12345",
          "name": "Wireless Bluetooth Headphones",
          "quantity": 2,
          "unit_price": 84.99,
          "subtotal": 169.98
        }
      ],
      "pricing": {
        "subtotal": 169.98,
        "tax": 15.30,
        "shipping": 0.00,
        "total": 185.28
      },
      "payment": {
        "status": "completed",
        "transaction_id": "txn_123456",
        "amount": 185.28
      },
      "shipping_address": {
        "name": "John Doe",
        "street": "123 Main St",
        "city": "Seattle",
        "state": "WA",
        "zip": "98101"
      },
      "estimated_delivery": "2026-02-05T23:59:59Z"
    }
    ```

    **Design Notes:**

    - Idempotency key prevents duplicate orders
    - Transaction: reserve inventory → charge payment → create order
    - Return immediately after order creation (async processing)
    - Send confirmation email asynchronously

    ---

    ### 4. Search Products

    **Request:**
    ```http
    GET /api/v1/search?q=wireless+headphones&category=electronics&
        min_price=50&max_price=200&rating=4&sort=popularity&page=1&size=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "query": "wireless headphones",
      "total_results": 1523,
      "page": 1,
      "page_size": 20,
      "filters_applied": {
        "category": "electronics",
        "price_range": [50, 200],
        "min_rating": 4
      },
      "products": [
        {
          "product_id": "12345",
          "name": "Wireless Bluetooth Headphones",
          "brand": "TechBrand",
          "price": 84.99,
          "original_price": 99.99,
          "rating": 4.5,
          "reviews_count": 1823,
          "image": "https://cdn.shop.com/products/12345/thumb.jpg",
          "in_stock": true,
          "prime_eligible": true
        },
        // ... 19 more products
      ],
      "facets": {
        "brands": [
          {"name": "TechBrand", "count": 234},
          {"name": "AudioPro", "count": 189}
        ],
        "price_ranges": [
          {"range": "0-50", "count": 45},
          {"range": "50-100", "count": 567}
        ]
      }
    }
    ```

    **Design Notes:**

    - Elasticsearch for fast full-text search
    - Faceted search for filtering (brands, price ranges)
    - Cache popular searches for 5 minutes
    - Autocomplete as separate endpoint

    ---

    ## Database Schema

    ### Products (PostgreSQL)

    ```sql
    -- Products table (with read replicas)
    CREATE TABLE products (
        product_id BIGINT PRIMARY KEY,
        name VARCHAR(500) NOT NULL,
        description TEXT,
        brand VARCHAR(200),
        category_id INT REFERENCES categories(category_id),
        base_price DECIMAL(10, 2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',

        -- Metadata
        specifications JSONB,
        dimensions JSONB,
        weight_grams INT,

        -- SEO
        slug VARCHAR(500) UNIQUE,
        meta_title VARCHAR(200),
        meta_description TEXT,

        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        -- Indexes
        INDEX idx_category (category_id),
        INDEX idx_brand (brand),
        INDEX idx_price (base_price),
        FULLTEXT INDEX idx_search (name, description)
    );

    -- Product images
    CREATE TABLE product_images (
        image_id BIGINT PRIMARY KEY,
        product_id BIGINT REFERENCES products(product_id),
        url TEXT NOT NULL,
        alt_text VARCHAR(200),
        display_order INT,
        is_primary BOOLEAN DEFAULT FALSE,

        INDEX idx_product (product_id)
    );

    -- Categories (hierarchical)
    CREATE TABLE categories (
        category_id INT PRIMARY KEY,
        parent_id INT REFERENCES categories(category_id),
        name VARCHAR(200) NOT NULL,
        slug VARCHAR(200) UNIQUE,
        level INT,
        path VARCHAR(500),  -- e.g., "Electronics/Audio/Headphones"

        INDEX idx_parent (parent_id),
        INDEX idx_path (path)
    );
    ```

    **Why PostgreSQL:**

    - **Rich data types:** JSONB for flexible product specs
    - **Read replicas:** Scale reads for catalog browsing
    - **Strong consistency:** Price updates must be consistent
    - **Full-text search:** Basic search capability (ES for advanced)

    ---

    ### Orders (PostgreSQL - Sharded by user_id)

    ```sql
    -- Orders table (ACID critical)
    CREATE TABLE orders (
        order_id BIGINT PRIMARY KEY,
        user_id BIGINT NOT NULL,
        status VARCHAR(50) NOT NULL,  -- pending, processing, shipped, delivered, cancelled

        -- Pricing
        subtotal DECIMAL(10, 2) NOT NULL,
        tax DECIMAL(10, 2) NOT NULL,
        shipping_cost DECIMAL(10, 2) NOT NULL,
        discount DECIMAL(10, 2) DEFAULT 0.00,
        total DECIMAL(10, 2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',

        -- Addresses (denormalized for immutability)
        shipping_address JSONB NOT NULL,
        billing_address JSONB NOT NULL,

        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        shipped_at TIMESTAMP,
        delivered_at TIMESTAMP,

        -- Idempotency
        idempotency_key VARCHAR(100) UNIQUE,

        INDEX idx_user_created (user_id, created_at DESC),
        INDEX idx_status (status),
        INDEX idx_idempotency (idempotency_key)
    ) PARTITION BY HASH (user_id);

    -- Order items
    CREATE TABLE order_items (
        order_item_id BIGINT PRIMARY KEY,
        order_id BIGINT REFERENCES orders(order_id),
        product_id BIGINT NOT NULL,

        -- Snapshot at purchase time
        product_name VARCHAR(500) NOT NULL,
        product_image TEXT,

        quantity INT NOT NULL CHECK (quantity > 0),
        unit_price DECIMAL(10, 2) NOT NULL,
        subtotal DECIMAL(10, 2) NOT NULL,

        -- Options (e.g., size, color)
        selected_options JSONB,

        INDEX idx_order (order_id),
        INDEX idx_product (product_id)
    );

    -- Payment transactions
    CREATE TABLE payment_transactions (
        transaction_id BIGINT PRIMARY KEY,
        order_id BIGINT REFERENCES orders(order_id),

        payment_method VARCHAR(50) NOT NULL,  -- credit_card, paypal, etc.
        payment_provider VARCHAR(50),  -- stripe, paypal
        provider_transaction_id VARCHAR(200),

        amount DECIMAL(10, 2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',
        status VARCHAR(50) NOT NULL,  -- pending, completed, failed, refunded

        -- Idempotency
        idempotency_key VARCHAR(100) UNIQUE,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_order (order_id),
        INDEX idx_status (status),
        INDEX idx_idempotency (idempotency_key)
    );
    ```

    **Why Sharded by user_id:**

    - **Write distribution:** Orders spread across shards
    - **Query locality:** User's order history on one shard
    - **Scalability:** Add shards as order volume grows

    ---

    ### Inventory (PostgreSQL - Strong Consistency)

    ```sql
    -- Inventory table (strong consistency critical)
    CREATE TABLE inventory (
        product_id BIGINT PRIMARY KEY,

        -- Stock levels
        total_quantity INT NOT NULL CHECK (total_quantity >= 0),
        reserved_quantity INT NOT NULL DEFAULT 0 CHECK (reserved_quantity >= 0),
        available_quantity INT GENERATED ALWAYS AS (total_quantity - reserved_quantity) STORED,

        -- Location
        warehouse_id VARCHAR(50),

        -- Thresholds
        low_stock_threshold INT DEFAULT 10,

        -- Timestamps
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_restocked TIMESTAMP,

        INDEX idx_warehouse (warehouse_id),
        INDEX idx_available (available_quantity)
    );

    -- Inventory reservations (for cart checkout)
    CREATE TABLE inventory_reservations (
        reservation_id BIGINT PRIMARY KEY,
        product_id BIGINT REFERENCES inventory(product_id),
        user_id BIGINT NOT NULL,
        quantity INT NOT NULL,

        status VARCHAR(50) DEFAULT 'active',  -- active, consumed, expired

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,  -- 10 minute TTL

        INDEX idx_product (product_id),
        INDEX idx_user (user_id),
        INDEX idx_expires (expires_at)
    );

    -- Inventory audit log
    CREATE TABLE inventory_audit_log (
        log_id BIGINT PRIMARY KEY,
        product_id BIGINT NOT NULL,

        action VARCHAR(50) NOT NULL,  -- restock, reserve, release, consume
        quantity_change INT NOT NULL,

        before_quantity INT NOT NULL,
        after_quantity INT NOT NULL,

        reason VARCHAR(200),
        performed_by VARCHAR(100),

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_product_time (product_id, created_at DESC)
    );
    ```

    **Why Strong Consistency:**

    - **No overselling:** Must prevent negative inventory
    - **ACID transactions:** Reserve → consume must be atomic
    - **Audit trail:** Track every inventory change

    ---

    ### Shopping Cart (Redis)

    ```redis
    # Cart stored as Redis hash with TTL
    # Key: cart:{user_id}
    # TTL: 30 days

    HSET cart:user123 product:12345 '{"quantity": 2, "price": 84.99, "added_at": "2026-02-02T10:30:00Z"}'
    HSET cart:user123 product:67890 '{"quantity": 1, "price": 29.99, "added_at": "2026-02-02T11:00:00Z"}'

    EXPIRE cart:user123 2592000  # 30 days

    # Get entire cart
    HGETALL cart:user123

    # Remove item
    HDEL cart:user123 product:12345

    # Cart metadata
    SET cart:user123:meta '{"created_at": "2026-02-02T10:30:00Z", "last_modified": "2026-02-02T11:00:00Z"}'
    ```

    **Why Redis:**

    - **Speed:** <1ms cart operations
    - **TTL:** Auto-expire abandoned carts
    - **Data structures:** Hash maps perfect for cart items
    - **High availability:** Redis Cluster with replication

    ---

    ### Users (PostgreSQL - Sharded)

    ```sql
    -- Users table (sharded by user_id)
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,

        first_name VARCHAR(100),
        last_name VARCHAR(100),
        phone VARCHAR(20),

        -- Preferences
        preferences JSONB,

        -- Status
        email_verified BOOLEAN DEFAULT FALSE,
        account_status VARCHAR(50) DEFAULT 'active',

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,

        INDEX idx_email (email)
    ) PARTITION BY HASH (user_id);

    -- Addresses
    CREATE TABLE user_addresses (
        address_id BIGINT PRIMARY KEY,
        user_id BIGINT REFERENCES users(user_id),

        name VARCHAR(200),
        street VARCHAR(500),
        city VARCHAR(100),
        state VARCHAR(100),
        zip VARCHAR(20),
        country VARCHAR(2),

        is_default BOOLEAN DEFAULT FALSE,
        address_type VARCHAR(50),  -- shipping, billing, both

        INDEX idx_user (user_id)
    );

    -- Payment methods
    CREATE TABLE payment_methods (
        payment_method_id BIGINT PRIMARY KEY,
        user_id BIGINT REFERENCES users(user_id),

        method_type VARCHAR(50),  -- credit_card, paypal, etc.

        -- Tokenized (never store raw card data)
        payment_token VARCHAR(200) NOT NULL,

        -- Display info
        last_four VARCHAR(4),
        brand VARCHAR(50),  -- Visa, Mastercard, etc.
        expiry_month INT,
        expiry_year INT,

        is_default BOOLEAN DEFAULT FALSE,

        INDEX idx_user (user_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Checkout Flow (Critical Path)

    ```mermaid
    sequenceDiagram
        participant Client
        participant Order_API
        participant Inventory_API
        participant Payment_API
        participant Order_DB
        participant Inventory_DB
        participant Payment_Gateway
        participant Kafka

        Client->>Order_API: POST /api/v1/orders (with idempotency key)
        Order_API->>Order_API: Check idempotency key (prevent duplicate)

        Note over Order_API: BEGIN TRANSACTION

        Order_API->>Inventory_API: Reserve inventory for cart items

        Inventory_API->>Inventory_DB: SELECT FOR UPDATE (lock rows)
        Inventory_DB-->>Inventory_API: Current inventory levels

        alt Sufficient inventory
            Inventory_API->>Inventory_DB: UPDATE inventory SET reserved_quantity += X
            Inventory_DB-->>Inventory_API: Reservation successful

            Inventory_API-->>Order_API: Inventory reserved (reservation_id)

            Order_API->>Payment_API: Charge payment
            Payment_API->>Payment_Gateway: Process payment

            alt Payment successful
                Payment_Gateway-->>Payment_API: Payment confirmed (txn_id)
                Payment_API->>Order_DB: INSERT payment_transaction (status=completed)
                Payment_API-->>Order_API: Payment successful

                Order_API->>Order_DB: INSERT order (status=processing)
                Order_API->>Order_DB: INSERT order_items

                Order_API->>Inventory_API: Consume reservation
                Inventory_API->>Inventory_DB: UPDATE inventory SET total_quantity -= X, reserved_quantity -= X

                Note over Order_API: COMMIT TRANSACTION

                Order_API->>Kafka: Publish order_created event
                Order_API-->>Client: 201 Created (order_id)

                Kafka->>Email_Service: Send order confirmation
                Kafka->>Analytics: Update sales metrics

            else Payment failed
                Payment_Gateway-->>Payment_API: Payment declined
                Payment_API-->>Order_API: Payment failed

                Order_API->>Inventory_API: Release reservation
                Inventory_API->>Inventory_DB: UPDATE inventory SET reserved_quantity -= X

                Note over Order_API: ROLLBACK TRANSACTION

                Order_API-->>Client: 402 Payment Required
            end

        else Insufficient inventory
            Inventory_DB-->>Inventory_API: Not enough stock
            Inventory_API-->>Order_API: Reservation failed (out of stock)

            Note over Order_API: ROLLBACK TRANSACTION

            Order_API-->>Client: 409 Conflict (out of stock)
        end
    ```

    **Flow Explanation:**

    1. **Idempotency check** - Prevent duplicate orders from retries
    2. **Inventory reservation** - Lock inventory with SELECT FOR UPDATE
    3. **Payment processing** - Charge customer via payment gateway
    4. **Order creation** - Insert order records in database
    5. **Inventory consumption** - Deduct from total inventory
    6. **Transaction commit** - All-or-nothing ACID transaction
    7. **Event publishing** - Async processing via Kafka

    **Critical:** Entire checkout is a single distributed transaction. If any step fails, rollback everything.

    ---

    ### Add to Cart Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Cart_API
        participant Inventory_API
        participant Redis_Cart
        participant Redis_Inventory
        participant Inventory_DB

        Client->>Cart_API: POST /api/v1/cart/items {product_id, quantity}

        Cart_API->>Inventory_API: Check availability

        Inventory_API->>Redis_Inventory: GET inventory:12345

        alt Cache HIT
            Redis_Inventory-->>Inventory_API: {available: 42}
        else Cache MISS
            Redis_Inventory-->>Inventory_API: null
            Inventory_API->>Inventory_DB: SELECT available_quantity
            Inventory_DB-->>Inventory_API: 42
            Inventory_API->>Redis_Inventory: SET inventory:12345 {available: 42} EX 60
        end

        alt In stock
            Inventory_API-->>Cart_API: Available (42 units)

            Cart_API->>Redis_Cart: HGET cart:user123 product:12345

            alt Item exists in cart
                Redis_Cart-->>Cart_API: {quantity: 1, price: 84.99}
                Cart_API->>Cart_API: Update quantity: 1 + 2 = 3
            else New item
                Redis_Cart-->>Cart_API: null
                Cart_API->>Cart_API: Create cart item: quantity = 2
            end

            Cart_API->>Redis_Cart: HSET cart:user123 product:12345 {quantity: 3, price: 84.99}
            Cart_API->>Redis_Cart: EXPIRE cart:user123 2592000

            Cart_API->>Redis_Cart: HGETALL cart:user123
            Redis_Cart-->>Cart_API: All cart items

            Cart_API-->>Client: 201 Created (full cart)

        else Out of stock
            Inventory_API-->>Cart_API: Out of stock
            Cart_API-->>Client: 409 Conflict (out of stock)
        end
    ```

    **Flow Explanation:**

    1. **Inventory check** - Verify product is in stock
    2. **Cache lookup** - Check Redis for fast response
    3. **Cart update** - Add or update quantity in Redis
    4. **TTL reset** - Extend cart expiry to 30 days
    5. **Return full cart** - Send complete cart for UI update

    ---

    ### Product Search Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Search_API
        participant Redis_Search
        participant Elasticsearch
        participant Product_DB
        participant Redis_Product

        Client->>Search_API: GET /api/v1/search?q=wireless+headphones

        Search_API->>Search_API: Generate cache key (query + filters)
        Search_API->>Redis_Search: GET search:{cache_key}

        alt Cache HIT (popular search)
            Redis_Search-->>Search_API: Cached results
            Search_API-->>Client: 200 OK (search results)
        else Cache MISS
            Redis_Search-->>Search_API: null

            Search_API->>Elasticsearch: Search query with filters
            Elasticsearch-->>Search_API: Product IDs + scores [12345, 67890, ...]

            Search_API->>Redis_Product: MGET product:12345 product:67890 ...
            Redis_Product-->>Search_API: Partial results (cache hits)

            Search_API->>Product_DB: SELECT * FROM products WHERE id IN (missing_ids)
            Product_DB-->>Search_API: Product details

            Search_API->>Redis_Product: MSET product:X ... (cache misses)

            Search_API->>Search_API: Merge and rank results

            Search_API->>Redis_Search: SET search:{cache_key} results EX 300

            Search_API-->>Client: 200 OK (search results)
        end
    ```

    **Flow Explanation:**

    1. **Cache popular searches** - Reduce ES load for common queries
    2. **Elasticsearch query** - Full-text search with filters
    3. **Hydrate products** - Fetch full product details
    4. **Multi-level caching** - Search results + product details
    5. **Cache for 5 minutes** - Balance freshness and performance

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical e-commerce subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Inventory Consistency** | How to prevent overselling with high concurrency? | Pessimistic locking + reservation system |
    | **Cart Management** | How to handle 10M concurrent carts efficiently? | Redis with TTL + cart persistence strategy |
    | **Payment Processing** | How to ensure idempotent, reliable payments? | Idempotency keys + distributed transactions |
    | **Flash Sales** | How to handle 10x traffic spikes? | Queue-based fairness + inventory pooling |

    ---

    === "🔒 Inventory Consistency"

        ## The Challenge

        **Problem:** Multiple users attempting to buy last item simultaneously. Must prevent overselling while maintaining performance.

        **Scenario:**
        - Product has 1 unit in stock
        - 100 users click "Buy Now" at exact same time
        - Only 1 should succeed, 99 should see "Out of Stock"

        **Naive approach:** Read inventory → check if available → decrease quantity. **Race condition!** Multiple transactions can read same value before any write.

        ---

        ## Solution: Pessimistic Locking with Reservations

        **Two-phase inventory management:**

        1. **Reservation phase:** Reserve inventory when user starts checkout (10 min TTL)
        2. **Consumption phase:** Consume reservation when payment succeeds

        **Benefits:**
        - Prevents overselling (strong consistency)
        - Allows cart abandonment (auto-release via TTL)
        - Fair allocation (first to checkout reserves)

        ---

        ## Implementation

        ```python
        from sqlalchemy import create_engine, select, update
        from sqlalchemy.orm import Session
        from datetime import datetime, timedelta
        import time

        class InventoryService:
            """
            Inventory management with strong consistency guarantees

            Uses pessimistic locking to prevent overselling
            """

            def __init__(self, db_engine, redis_client):
                self.db = db_engine
                self.redis = redis_client
                self.RESERVATION_TTL = 600  # 10 minutes

            def check_availability(self, product_id: int, quantity: int) -> bool:
                """
                Check if product has sufficient inventory (cached)

                Args:
                    product_id: Product to check
                    quantity: Quantity needed

                Returns:
                    True if available, False otherwise
                """
                # Check cache first (eventual consistency OK for display)
                cache_key = f"inventory:{product_id}"
                cached = self.redis.get(cache_key)

                if cached:
                    available = int(cached)
                    return available >= quantity

                # Cache miss - query database
                with Session(self.db) as session:
                    inventory = session.execute(
                        select(Inventory.available_quantity)
                        .where(Inventory.product_id == product_id)
                    ).scalar_one_or_none()

                    if inventory is None:
                        return False

                    # Cache for 60 seconds
                    self.redis.setex(cache_key, 60, inventory)

                    return inventory >= quantity

            def reserve_inventory(
                self,
                product_id: int,
                quantity: int,
                user_id: int
            ) -> dict:
                """
                Reserve inventory for checkout (strong consistency)

                Uses SELECT FOR UPDATE to prevent race conditions

                Args:
                    product_id: Product to reserve
                    quantity: Quantity to reserve
                    user_id: User making reservation

                Returns:
                    Reservation details or error
                """
                with Session(self.db) as session:
                    try:
                        # BEGIN TRANSACTION (implicit)

                        # Pessimistic lock: SELECT FOR UPDATE
                        inventory = session.execute(
                            select(Inventory)
                            .where(Inventory.product_id == product_id)
                            .with_for_update()  # Row-level lock
                        ).scalar_one_or_none()

                        if not inventory:
                            return {'success': False, 'error': 'product_not_found'}

                        # Check available quantity
                        if inventory.available_quantity < quantity:
                            return {
                                'success': False,
                                'error': 'insufficient_inventory',
                                'available': inventory.available_quantity
                            }

                        # Create reservation
                        reservation = InventoryReservation(
                            product_id=product_id,
                            user_id=user_id,
                            quantity=quantity,
                            status='active',
                            created_at=datetime.utcnow(),
                            expires_at=datetime.utcnow() + timedelta(seconds=self.RESERVATION_TTL)
                        )
                        session.add(reservation)

                        # Update inventory (increase reserved_quantity)
                        session.execute(
                            update(Inventory)
                            .where(Inventory.product_id == product_id)
                            .values(reserved_quantity=Inventory.reserved_quantity + quantity)
                        )

                        # Audit log
                        audit = InventoryAuditLog(
                            product_id=product_id,
                            action='reserve',
                            quantity_change=quantity,
                            before_quantity=inventory.available_quantity,
                            after_quantity=inventory.available_quantity - quantity,
                            reason=f'Reservation for user {user_id}',
                            performed_by=f'user:{user_id}'
                        )
                        session.add(audit)

                        # COMMIT
                        session.commit()

                        # Invalidate cache
                        self.redis.delete(f"inventory:{product_id}")

                        return {
                            'success': True,
                            'reservation_id': reservation.reservation_id,
                            'expires_at': reservation.expires_at.isoformat()
                        }

                    except Exception as e:
                        # ROLLBACK (automatic)
                        session.rollback()
                        logger.error(f"Reservation failed: {e}")
                        return {'success': False, 'error': 'database_error'}

            def consume_reservation(self, reservation_id: int) -> bool:
                """
                Consume reservation after successful payment

                Decreases total_quantity and reserved_quantity

                Args:
                    reservation_id: Reservation to consume

                Returns:
                    True if successful, False otherwise
                """
                with Session(self.db) as session:
                    try:
                        # Lock reservation
                        reservation = session.execute(
                            select(InventoryReservation)
                            .where(InventoryReservation.reservation_id == reservation_id)
                            .with_for_update()
                        ).scalar_one_or_none()

                        if not reservation or reservation.status != 'active':
                            return False

                        # Check expiry
                        if datetime.utcnow() > reservation.expires_at:
                            reservation.status = 'expired'
                            session.commit()
                            return False

                        # Lock inventory
                        inventory = session.execute(
                            select(Inventory)
                            .where(Inventory.product_id == reservation.product_id)
                            .with_for_update()
                        ).scalar_one()

                        # Consume: decrease both total and reserved
                        session.execute(
                            update(Inventory)
                            .where(Inventory.product_id == reservation.product_id)
                            .values(
                                total_quantity=Inventory.total_quantity - reservation.quantity,
                                reserved_quantity=Inventory.reserved_quantity - reservation.quantity
                            )
                        )

                        # Mark reservation as consumed
                        reservation.status = 'consumed'

                        # Audit log
                        audit = InventoryAuditLog(
                            product_id=reservation.product_id,
                            action='consume',
                            quantity_change=-reservation.quantity,
                            before_quantity=inventory.total_quantity,
                            after_quantity=inventory.total_quantity - reservation.quantity,
                            reason=f'Order placed, reservation {reservation_id}',
                            performed_by=f'user:{reservation.user_id}'
                        )
                        session.add(audit)

                        session.commit()

                        # Invalidate cache
                        self.redis.delete(f"inventory:{reservation.product_id}")

                        return True

                    except Exception as e:
                        session.rollback()
                        logger.error(f"Consume failed: {e}")
                        return False

            def release_reservation(self, reservation_id: int) -> bool:
                """
                Release reservation (cart abandonment or payment failure)

                Returns inventory to available pool

                Args:
                    reservation_id: Reservation to release

                Returns:
                    True if successful
                """
                with Session(self.db) as session:
                    try:
                        reservation = session.execute(
                            select(InventoryReservation)
                            .where(InventoryReservation.reservation_id == reservation_id)
                            .with_for_update()
                        ).scalar_one_or_none()

                        if not reservation or reservation.status != 'active':
                            return False

                        # Decrease reserved_quantity (total_quantity unchanged)
                        session.execute(
                            update(Inventory)
                            .where(Inventory.product_id == reservation.product_id)
                            .values(
                                reserved_quantity=Inventory.reserved_quantity - reservation.quantity
                            )
                        )

                        reservation.status = 'released'

                        # Audit log
                        audit = InventoryAuditLog(
                            product_id=reservation.product_id,
                            action='release',
                            quantity_change=reservation.quantity,
                            before_quantity=0,  # Reserved quantity released
                            after_quantity=reservation.quantity,
                            reason=f'Reservation released {reservation_id}',
                            performed_by='system'
                        )
                        session.add(audit)

                        session.commit()

                        # Invalidate cache
                        self.redis.delete(f"inventory:{reservation.product_id}")

                        return True

                    except Exception as e:
                        session.rollback()
                        logger.error(f"Release failed: {e}")
                        return False

            def cleanup_expired_reservations(self):
                """
                Background job: Release expired reservations

                Run every 1 minute via cron
                """
                with Session(self.db) as session:
                    # Find expired active reservations
                    expired = session.execute(
                        select(InventoryReservation)
                        .where(
                            InventoryReservation.status == 'active',
                            InventoryReservation.expires_at < datetime.utcnow()
                        )
                    ).scalars().all()

                    logger.info(f"Found {len(expired)} expired reservations")

                    for reservation in expired:
                        self.release_reservation(reservation.reservation_id)
        ```

        ---

        ## Concurrency Handling

        **Key mechanism: SELECT FOR UPDATE**

        ```sql
        -- Transaction 1 (User A)
        BEGIN;
        SELECT * FROM inventory WHERE product_id = 12345 FOR UPDATE;
        -- Acquires row lock, other transactions wait
        UPDATE inventory SET reserved_quantity = reserved_quantity + 1 WHERE product_id = 12345;
        COMMIT;
        -- Releases lock

        -- Transaction 2 (User B)
        BEGIN;
        SELECT * FROM inventory WHERE product_id = 12345 FOR UPDATE;
        -- Waits for Transaction 1 to commit
        -- Then proceeds with up-to-date data
        ```

        **Benefits:**
        - **No race conditions:** Only one transaction modifies inventory at a time
        - **Fair ordering:** FIFO queue for lock acquisition
        - **Automatic deadlock detection:** PostgreSQL handles deadlocks

        **Trade-off:**
        - **Performance:** Locks reduce concurrency (but correctness > speed)
        - **Latency:** Transactions may wait for locks (typically <10ms)

        ---

        ## Alternative: Optimistic Locking

        **When to use:**
        - Low contention scenarios
        - Non-critical inventory (digital goods)

        ```python
        def reserve_inventory_optimistic(self, product_id: int, quantity: int):
            """
            Optimistic locking using version numbers

            Retry on conflict (no blocking)
            """
            max_retries = 3

            for attempt in range(max_retries):
                with Session(self.db) as session:
                    # Read current state
                    inventory = session.execute(
                        select(Inventory)
                        .where(Inventory.product_id == product_id)
                    ).scalar_one()

                    version = inventory.version

                    if inventory.available_quantity < quantity:
                        return {'success': False, 'error': 'insufficient_inventory'}

                    # Try to update with version check
                    result = session.execute(
                        update(Inventory)
                        .where(
                            Inventory.product_id == product_id,
                            Inventory.version == version  # Optimistic lock
                        )
                        .values(
                            reserved_quantity=Inventory.reserved_quantity + quantity,
                            version=Inventory.version + 1
                        )
                    )

                    if result.rowcount == 1:
                        # Success - version matched
                        session.commit()
                        return {'success': True}
                    else:
                        # Version mismatch - retry
                        logger.warning(f"Optimistic lock conflict, retry {attempt+1}")
                        time.sleep(0.01 * (attempt + 1))  # Exponential backoff

            return {'success': False, 'error': 'max_retries_exceeded'}
        ```

        **Trade-offs:**

        | Approach | Pros | Cons | Use Case |
        |----------|------|------|----------|
        | **Pessimistic (SELECT FOR UPDATE)** | Guaranteed consistency, no retries | Lower concurrency, blocking | High contention (limited stock) |
        | **Optimistic (version check)** | Higher concurrency, no blocking | Retry logic, not guaranteed | Low contention (abundant stock) |

    === "🛒 Cart Management"

        ## The Challenge

        **Problem:** Handle 10M concurrent shopping carts with sub-second latency. Carts must persist across sessions and devices.

        **Requirements:**
        - **Fast:** <10ms cart operations (add, remove, update)
        - **Persistent:** Cart survives browser close, device switch
        - **TTL:** Auto-expire abandoned carts (30 days)
        - **Consistent:** Cart state synchronized across devices

        ---

        ## Solution: Redis + Database Hybrid

        **Strategy:**
        1. **Hot carts (Redis):** Active carts (last 24 hours) in Redis
        2. **Cold carts (PostgreSQL):** Inactive carts persisted to database
        3. **Lazy loading:** Load from DB to Redis on user activity

        ---

        ## Implementation

        ```python
        import json
        import redis
        from datetime import datetime, timedelta
        from sqlalchemy.orm import Session

        class CartService:
            """
            Shopping cart management with Redis + DB persistence

            Hot path: Redis (fast)
            Cold path: PostgreSQL (durable)
            """

            CART_TTL = 30 * 24 * 60 * 60  # 30 days
            CART_PERSIST_THRESHOLD = 24 * 60 * 60  # 24 hours

            def __init__(self, redis_client, db_engine, inventory_service):
                self.redis = redis_client
                self.db = db_engine
                self.inventory = inventory_service

            def get_cart(self, user_id: int) -> dict:
                """
                Get user's shopping cart

                Checks Redis first, falls back to database

                Args:
                    user_id: User ID

                Returns:
                    Cart with items and summary
                """
                cache_key = f"cart:{user_id}"

                # Try Redis first
                cart_data = self.redis.hgetall(cache_key)

                if cart_data:
                    # Cart in Redis (hot)
                    items = [
                        json.loads(item_json)
                        for item_json in cart_data.values()
                    ]

                    # Refresh TTL
                    self.redis.expire(cache_key, self.CART_TTL)

                    return self._build_cart_response(user_id, items)

                # Cache miss - check database
                with Session(self.db) as session:
                    db_cart = session.execute(
                        select(Cart).where(Cart.user_id == user_id)
                    ).scalar_one_or_none()

                    if not db_cart:
                        # No cart exists
                        return self._empty_cart(user_id)

                    # Load items from database
                    items = session.execute(
                        select(CartItem)
                        .where(CartItem.cart_id == db_cart.cart_id)
                    ).scalars().all()

                    # Hydrate to Redis for future requests
                    self._load_cart_to_redis(user_id, items)

                    return self._build_cart_response(user_id, items)

            def add_item(
                self,
                user_id: int,
                product_id: int,
                quantity: int,
                selected_options: dict = None
            ) -> dict:
                """
                Add item to cart

                Args:
                    user_id: User ID
                    product_id: Product to add
                    quantity: Quantity to add
                    selected_options: Product options (size, color, etc.)

                Returns:
                    Updated cart
                """
                # Validate inventory
                if not self.inventory.check_availability(product_id, quantity):
                    return {
                        'success': False,
                        'error': 'out_of_stock',
                        'product_id': product_id
                    }

                # Get product details (for price, name)
                product = self._get_product(product_id)
                if not product:
                    return {'success': False, 'error': 'product_not_found'}

                cache_key = f"cart:{user_id}"
                item_key = f"product:{product_id}"

                # Check if item already in cart
                existing_item = self.redis.hget(cache_key, item_key)

                if existing_item:
                    # Update quantity
                    item_data = json.loads(existing_item)
                    new_quantity = item_data['quantity'] + quantity

                    # Re-check inventory for new quantity
                    if not self.inventory.check_availability(product_id, new_quantity):
                        return {
                            'success': False,
                            'error': 'insufficient_inventory',
                            'available': self.inventory.check_availability(product_id, 0)
                        }

                    item_data['quantity'] = new_quantity
                    item_data['updated_at'] = datetime.utcnow().isoformat()
                else:
                    # New item
                    item_data = {
                        'product_id': product_id,
                        'name': product['name'],
                        'quantity': quantity,
                        'unit_price': product['price'],
                        'image': product['image'],
                        'selected_options': selected_options or {},
                        'added_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    }

                # Save to Redis
                self.redis.hset(cache_key, item_key, json.dumps(item_data))
                self.redis.expire(cache_key, self.CART_TTL)

                # Update cart metadata
                meta_key = f"cart:{user_id}:meta"
                self.redis.setex(
                    meta_key,
                    self.CART_TTL,
                    json.dumps({
                        'user_id': user_id,
                        'last_modified': datetime.utcnow().isoformat()
                    })
                )

                # Get full cart
                cart = self.get_cart(user_id)

                return {
                    'success': True,
                    'cart': cart
                }

            def remove_item(self, user_id: int, product_id: int) -> dict:
                """
                Remove item from cart

                Args:
                    user_id: User ID
                    product_id: Product to remove

                Returns:
                    Updated cart
                """
                cache_key = f"cart:{user_id}"
                item_key = f"product:{product_id}"

                # Remove from Redis
                self.redis.hdel(cache_key, item_key)

                # Update metadata
                meta_key = f"cart:{user_id}:meta"
                self.redis.setex(
                    meta_key,
                    self.CART_TTL,
                    json.dumps({
                        'user_id': user_id,
                        'last_modified': datetime.utcnow().isoformat()
                    })
                )

                cart = self.get_cart(user_id)

                return {
                    'success': True,
                    'cart': cart
                }

            def update_quantity(
                self,
                user_id: int,
                product_id: int,
                new_quantity: int
            ) -> dict:
                """
                Update item quantity in cart

                Args:
                    user_id: User ID
                    product_id: Product to update
                    new_quantity: New quantity (0 = remove)

                Returns:
                    Updated cart
                """
                if new_quantity == 0:
                    return self.remove_item(user_id, product_id)

                # Check inventory
                if not self.inventory.check_availability(product_id, new_quantity):
                    return {'success': False, 'error': 'insufficient_inventory'}

                cache_key = f"cart:{user_id}"
                item_key = f"product:{product_id}"

                # Get existing item
                item_json = self.redis.hget(cache_key, item_key)
                if not item_json:
                    return {'success': False, 'error': 'item_not_in_cart'}

                # Update quantity
                item_data = json.loads(item_json)
                item_data['quantity'] = new_quantity
                item_data['updated_at'] = datetime.utcnow().isoformat()

                # Save
                self.redis.hset(cache_key, item_key, json.dumps(item_data))

                cart = self.get_cart(user_id)

                return {
                    'success': True,
                    'cart': cart
                }

            def clear_cart(self, user_id: int):
                """
                Clear user's cart

                Called after successful order placement

                Args:
                    user_id: User ID
                """
                cache_key = f"cart:{user_id}"
                meta_key = f"cart:{user_id}:meta"

                # Delete from Redis
                self.redis.delete(cache_key)
                self.redis.delete(meta_key)

                # Delete from database
                with Session(self.db) as session:
                    cart = session.execute(
                        select(Cart).where(Cart.user_id == user_id)
                    ).scalar_one_or_none()

                    if cart:
                        session.execute(
                            delete(CartItem).where(CartItem.cart_id == cart.cart_id)
                        )
                        session.delete(cart)
                        session.commit()

            def persist_cart_to_db(self, user_id: int):
                """
                Persist cart from Redis to database

                Background job: Run for carts inactive > 24 hours

                Args:
                    user_id: User ID
                """
                cache_key = f"cart:{user_id}"

                # Get cart from Redis
                cart_data = self.redis.hgetall(cache_key)
                if not cart_data:
                    return

                with Session(self.db) as session:
                    # Get or create cart
                    cart = session.execute(
                        select(Cart).where(Cart.user_id == user_id)
                    ).scalar_one_or_none()

                    if not cart:
                        cart = Cart(user_id=user_id, created_at=datetime.utcnow())
                        session.add(cart)
                        session.flush()  # Get cart_id

                    # Delete existing items
                    session.execute(
                        delete(CartItem).where(CartItem.cart_id == cart.cart_id)
                    )

                    # Insert current items
                    for item_json in cart_data.values():
                        item = json.loads(item_json)

                        cart_item = CartItem(
                            cart_id=cart.cart_id,
                            product_id=item['product_id'],
                            quantity=item['quantity'],
                            unit_price=item['unit_price'],
                            selected_options=item.get('selected_options')
                        )
                        session.add(cart_item)

                    cart.updated_at = datetime.utcnow()
                    session.commit()

                logger.info(f"Persisted cart for user {user_id} to database")

            def _build_cart_response(self, user_id: int, items: list) -> dict:
                """
                Build cart response with summary

                Args:
                    user_id: User ID
                    items: Cart items

                Returns:
                    Cart object with items and totals
                """
                cart_items = []
                subtotal = 0.0

                for item in items:
                    item_subtotal = item['unit_price'] * item['quantity']
                    subtotal += item_subtotal

                    cart_items.append({
                        'cart_item_id': f"item_{item['product_id']}",
                        'product_id': item['product_id'],
                        'name': item['name'],
                        'quantity': item['quantity'],
                        'unit_price': item['unit_price'],
                        'subtotal': item_subtotal,
                        'image': item.get('image'),
                        'selected_options': item.get('selected_options', {})
                    })

                # Calculate tax and shipping (simplified)
                tax_rate = 0.09  # 9% sales tax
                tax = round(subtotal * tax_rate, 2)
                shipping = 0.0 if subtotal > 50 else 5.99  # Free shipping over $50
                total = round(subtotal + tax + shipping, 2)

                return {
                    'cart_id': f"cart_{user_id}",
                    'user_id': user_id,
                    'items': cart_items,
                    'summary': {
                        'items_count': sum(item['quantity'] for item in items),
                        'subtotal': round(subtotal, 2),
                        'tax': tax,
                        'shipping': shipping,
                        'total': total
                    },
                    'expires_at': (datetime.utcnow() + timedelta(seconds=self.CART_TTL)).isoformat()
                }

            def _empty_cart(self, user_id: int) -> dict:
                """Return empty cart structure"""
                return {
                    'cart_id': f"cart_{user_id}",
                    'user_id': user_id,
                    'items': [],
                    'summary': {
                        'items_count': 0,
                        'subtotal': 0.0,
                        'tax': 0.0,
                        'shipping': 0.0,
                        'total': 0.0
                    }
                }

            def _load_cart_to_redis(self, user_id: int, items: list):
                """Load cart from database to Redis"""
                cache_key = f"cart:{user_id}"

                for item in items:
                    item_key = f"product:{item.product_id}"
                    item_data = {
                        'product_id': item.product_id,
                        'name': item.name,
                        'quantity': item.quantity,
                        'unit_price': float(item.unit_price),
                        'image': item.image,
                        'selected_options': item.selected_options or {},
                        'added_at': item.created_at.isoformat(),
                        'updated_at': item.updated_at.isoformat()
                    }
                    self.redis.hset(cache_key, item_key, json.dumps(item_data))

                self.redis.expire(cache_key, self.CART_TTL)
        ```

        ---

        ## Cart Persistence Strategy

        **Background job (runs every hour):**

        ```python
        def persist_inactive_carts():
            """
            Background job: Persist inactive carts to database

            Run via cron: 0 * * * * (hourly)
            """
            # Find carts inactive for > 24 hours
            pattern = "cart:*"
            cursor = 0

            while True:
                cursor, keys = redis_client.scan(
                    cursor,
                    match=pattern,
                    count=1000
                )

                for key in keys:
                    user_id = int(key.decode().split(':')[1])

                    # Check last activity
                    meta_key = f"cart:{user_id}:meta"
                    meta = redis_client.get(meta_key)

                    if meta:
                        meta_data = json.loads(meta)
                        last_modified = datetime.fromisoformat(meta_data['last_modified'])

                        # Inactive for > 24 hours?
                        if datetime.utcnow() - last_modified > timedelta(hours=24):
                            # Persist to database
                            cart_service.persist_cart_to_db(user_id)

                            # Remove from Redis (save memory)
                            redis_client.delete(key)
                            redis_client.delete(meta_key)

                            logger.info(f"Persisted inactive cart for user {user_id}")

                if cursor == 0:
                    break
        ```

        ---

        ## Cart Sync Across Devices

        **Challenge:** User adds item on mobile, switches to desktop - cart must sync.

        **Solution:** Redis as single source of truth

        1. **Mobile app:** Adds item → writes to Redis
        2. **Desktop browser:** Loads page → reads from Redis
        3. **Real-time sync:** WebSocket pushes cart updates to all user's devices

        ```python
        # WebSocket: Cart update notification
        def notify_cart_update(user_id: int, cart: dict):
            """
            Push cart update to all user's connected devices

            Args:
                user_id: User to notify
                cart: Updated cart object
            """
            # Publish to Redis pub/sub
            redis_client.publish(
                f"user:{user_id}:cart_updates",
                json.dumps({
                    'type': 'cart_updated',
                    'cart': cart
                })
            )
        ```

    === "💳 Payment Processing"

        ## The Challenge

        **Problem:** Process 2M orders/day with zero payment errors. Must handle network failures, retries, and ensure exactly-once payment processing.

        **Critical requirements:**
        - **Idempotency:** Retry same request = no duplicate charges
        - **Atomicity:** Either full order succeeds or full rollback
        - **Audit trail:** Every payment logged for reconciliation
        - **Timeout handling:** Network failures don't lose money

        ---

        ## Solution: Idempotency Keys + Distributed Transactions

        **Key patterns:**

        1. **Idempotency keys:** Client-generated unique ID for each request
        2. **Two-phase commit:** Reserve → charge → fulfill
        3. **Compensation transactions:** Rollback on failure (refund)
        4. **Retry with exponential backoff:** Handle transient failures

        ---

        ## Implementation

        ```python
        import uuid
        import time
        from decimal import Decimal
        from sqlalchemy.orm import Session
        from stripe import Stripe

        class PaymentService:
            """
            Payment processing with idempotency and strong guarantees

            Integrates with payment gateways (Stripe, PayPal)
            """

            def __init__(self, db_engine, stripe_client, kafka_producer):
                self.db = db_engine
                self.stripe = stripe_client
                self.kafka = kafka_producer

            def process_payment(
                self,
                order_id: int,
                user_id: int,
                amount: Decimal,
                currency: str,
                payment_method_id: str,
                idempotency_key: str
            ) -> dict:
                """
                Process payment for order

                Uses idempotency key to prevent duplicate charges

                Args:
                    order_id: Order being paid for
                    user_id: User making payment
                    amount: Payment amount
                    currency: Currency code (USD, EUR, etc.)
                    payment_method_id: Saved payment method
                    idempotency_key: Client-generated unique key

                Returns:
                    Payment result with transaction ID
                """
                with Session(self.db) as session:
                    # Check if already processed (idempotency)
                    existing = session.execute(
                        select(PaymentTransaction)
                        .where(PaymentTransaction.idempotency_key == idempotency_key)
                    ).scalar_one_or_none()

                    if existing:
                        logger.info(f"Duplicate payment request: {idempotency_key}, returning existing")

                        return {
                            'success': existing.status == 'completed',
                            'transaction_id': existing.transaction_id,
                            'status': existing.status,
                            'amount': float(existing.amount),
                            'is_duplicate': True
                        }

                    # Create payment transaction record (pending)
                    transaction = PaymentTransaction(
                        order_id=order_id,
                        payment_method='credit_card',
                        payment_provider='stripe',
                        amount=amount,
                        currency=currency,
                        status='pending',
                        idempotency_key=idempotency_key,
                        created_at=datetime.utcnow()
                    )
                    session.add(transaction)
                    session.commit()  # Save pending state

                    transaction_id = transaction.transaction_id

                try:
                    # Charge via Stripe (with Stripe's idempotency)
                    stripe_response = self.stripe.charges.create(
                        amount=int(amount * 100),  # Convert to cents
                        currency=currency,
                        payment_method=payment_method_id,
                        customer=self._get_stripe_customer_id(user_id),
                        description=f"Order {order_id}",
                        idempotency_key=idempotency_key,  # Stripe's idempotency
                        capture=True  # Immediate capture
                    )

                    # Payment successful
                    with Session(self.db) as session:
                        transaction = session.get(PaymentTransaction, transaction_id)
                        transaction.status = 'completed'
                        transaction.provider_transaction_id = stripe_response.id
                        transaction.updated_at = datetime.utcnow()
                        session.commit()

                    # Publish success event
                    self.kafka.send(
                        'payment_events',
                        {
                            'event_type': 'payment_completed',
                            'transaction_id': transaction_id,
                            'order_id': order_id,
                            'amount': float(amount)
                        }
                    )

                    logger.info(f"Payment successful: {transaction_id}, Stripe: {stripe_response.id}")

                    return {
                        'success': True,
                        'transaction_id': transaction_id,
                        'provider_transaction_id': stripe_response.id,
                        'status': 'completed',
                        'amount': float(amount)
                    }

                except stripe.error.CardError as e:
                    # Card declined
                    logger.warning(f"Card declined: {e}")

                    with Session(self.db) as session:
                        transaction = session.get(PaymentTransaction, transaction_id)
                        transaction.status = 'failed'
                        transaction.failure_reason = str(e)
                        transaction.updated_at = datetime.utcnow()
                        session.commit()

                    return {
                        'success': False,
                        'transaction_id': transaction_id,
                        'status': 'failed',
                        'error': 'card_declined',
                        'message': str(e)
                    }

                except stripe.error.StripeError as e:
                    # Stripe API error (network, timeout, etc.)
                    logger.error(f"Stripe error: {e}")

                    with Session(self.db) as session:
                        transaction = session.get(PaymentTransaction, transaction_id)
                        transaction.status = 'error'
                        transaction.failure_reason = str(e)
                        transaction.updated_at = datetime.utcnow()
                        session.commit()

                    # This will trigger retry by caller
                    return {
                        'success': False,
                        'transaction_id': transaction_id,
                        'status': 'error',
                        'error': 'payment_gateway_error',
                        'message': 'Please try again',
                        'retryable': True
                    }

                except Exception as e:
                    # Unexpected error
                    logger.exception(f"Unexpected payment error: {e}")

                    with Session(self.db) as session:
                        transaction = session.get(PaymentTransaction, transaction_id)
                        transaction.status = 'error'
                        transaction.failure_reason = str(e)
                        transaction.updated_at = datetime.utcnow()
                        session.commit()

                    return {
                        'success': False,
                        'transaction_id': transaction_id,
                        'status': 'error',
                        'error': 'internal_error'
                    }

            def refund_payment(
                self,
                transaction_id: int,
                amount: Decimal = None,
                reason: str = None
            ) -> dict:
                """
                Refund a completed payment

                Args:
                    transaction_id: Transaction to refund
                    amount: Refund amount (None = full refund)
                    reason: Refund reason

                Returns:
                    Refund result
                """
                with Session(self.db) as session:
                    transaction = session.get(PaymentTransaction, transaction_id)

                    if not transaction or transaction.status != 'completed':
                        return {
                            'success': False,
                            'error': 'invalid_transaction'
                        }

                    # Default to full refund
                    refund_amount = amount or transaction.amount

                    try:
                        # Refund via Stripe
                        stripe_refund = self.stripe.refunds.create(
                            charge=transaction.provider_transaction_id,
                            amount=int(refund_amount * 100),
                            reason=reason or 'requested_by_customer'
                        )

                        # Update transaction
                        transaction.status = 'refunded'
                        transaction.refund_amount = refund_amount
                        transaction.refund_reason = reason
                        transaction.updated_at = datetime.utcnow()
                        session.commit()

                        # Publish event
                        self.kafka.send(
                            'payment_events',
                            {
                                'event_type': 'payment_refunded',
                                'transaction_id': transaction_id,
                                'order_id': transaction.order_id,
                                'amount': float(refund_amount)
                            }
                        )

                        logger.info(f"Refund successful: {transaction_id}, amount: {refund_amount}")

                        return {
                            'success': True,
                            'refund_id': stripe_refund.id,
                            'amount': float(refund_amount)
                        }

                    except Exception as e:
                        logger.exception(f"Refund failed: {e}")
                        return {
                            'success': False,
                            'error': 'refund_failed',
                            'message': str(e)
                        }

            def _get_stripe_customer_id(self, user_id: int) -> str:
                """
                Get or create Stripe customer ID for user

                Args:
                    user_id: User ID

                Returns:
                    Stripe customer ID
                """
                with Session(self.db) as session:
                    user = session.get(User, user_id)

                    if user.stripe_customer_id:
                        return user.stripe_customer_id

                    # Create Stripe customer
                    customer = self.stripe.customers.create(
                        email=user.email,
                        metadata={'user_id': user_id}
                    )

                    user.stripe_customer_id = customer.id
                    session.commit()

                    return customer.id
        ```

        ---

        ## Distributed Transaction: Order + Payment

        **Saga pattern for distributed transaction:**

        ```python
        class OrderService:
            """
            Order creation with distributed transaction

            Saga: Inventory → Payment → Order creation
            Compensations: Release inventory, refund payment on failure
            """

            def __init__(
                self,
                db_engine,
                inventory_service,
                payment_service,
                kafka_producer
            ):
                self.db = db_engine
                self.inventory = inventory_service
                self.payment = payment_service
                self.kafka = kafka_producer

            def create_order(
                self,
                user_id: int,
                cart: dict,
                shipping_address: dict,
                payment_method_id: str,
                idempotency_key: str
            ) -> dict:
                """
                Create order with distributed transaction

                Saga steps:
                1. Reserve inventory
                2. Process payment
                3. Create order
                4. Consume inventory
                5. Clear cart

                On failure: Rollback all steps

                Args:
                    user_id: User placing order
                    cart: Shopping cart
                    shipping_address: Delivery address
                    payment_method_id: Payment method
                    idempotency_key: Prevent duplicate orders

                Returns:
                    Order result
                """
                # Check for duplicate order (idempotency)
                with Session(self.db) as session:
                    existing = session.execute(
                        select(Order)
                        .where(Order.idempotency_key == idempotency_key)
                    ).scalar_one_or_none()

                    if existing:
                        logger.info(f"Duplicate order request: {idempotency_key}")
                        return {
                            'success': True,
                            'order_id': existing.order_id,
                            'is_duplicate': True
                        }

                # State for rollback
                reservations = []
                payment_transaction_id = None
                order_id = None

                try:
                    # STEP 1: Reserve inventory for all items
                    logger.info(f"Step 1: Reserving inventory for {len(cart['items'])} items")

                    for item in cart['items']:
                        reservation_result = self.inventory.reserve_inventory(
                            product_id=item['product_id'],
                            quantity=item['quantity'],
                            user_id=user_id
                        )

                        if not reservation_result['success']:
                            # Inventory reservation failed
                            logger.warning(f"Inventory reservation failed: {reservation_result}")
                            raise InventoryReservationError(
                                f"Product {item['product_id']} - {reservation_result['error']}"
                            )

                        reservations.append(reservation_result['reservation_id'])

                    logger.info(f"Inventory reserved: {len(reservations)} reservations")

                    # STEP 2: Process payment
                    logger.info(f"Step 2: Processing payment: {cart['summary']['total']}")

                    payment_result = self.payment.process_payment(
                        order_id=0,  # Not created yet
                        user_id=user_id,
                        amount=Decimal(str(cart['summary']['total'])),
                        currency='USD',
                        payment_method_id=payment_method_id,
                        idempotency_key=f"{idempotency_key}_payment"
                    )

                    if not payment_result['success']:
                        # Payment failed
                        logger.warning(f"Payment failed: {payment_result}")
                        raise PaymentFailedError(payment_result.get('message', 'Payment declined'))

                    payment_transaction_id = payment_result['transaction_id']
                    logger.info(f"Payment successful: {payment_transaction_id}")

                    # STEP 3: Create order
                    logger.info(f"Step 3: Creating order record")

                    with Session(self.db) as session:
                        order = Order(
                            user_id=user_id,
                            status='processing',
                            subtotal=cart['summary']['subtotal'],
                            tax=cart['summary']['tax'],
                            shipping_cost=cart['summary']['shipping'],
                            total=cart['summary']['total'],
                            currency='USD',
                            shipping_address=shipping_address,
                            billing_address=shipping_address,  # Same for simplicity
                            idempotency_key=idempotency_key,
                            created_at=datetime.utcnow()
                        )
                        session.add(order)
                        session.flush()  # Get order_id

                        order_id = order.order_id

                        # Create order items
                        for item in cart['items']:
                            order_item = OrderItem(
                                order_id=order_id,
                                product_id=item['product_id'],
                                product_name=item['name'],
                                product_image=item['image'],
                                quantity=item['quantity'],
                                unit_price=item['unit_price'],
                                subtotal=item['subtotal'],
                                selected_options=item.get('selected_options')
                            )
                            session.add(order_item)

                        session.commit()

                    logger.info(f"Order created: {order_id}")

                    # STEP 4: Consume inventory reservations
                    logger.info(f"Step 4: Consuming inventory reservations")

                    for reservation_id in reservations:
                        success = self.inventory.consume_reservation(reservation_id)
                        if not success:
                            logger.error(f"Failed to consume reservation {reservation_id}")
                            # Continue anyway - inventory already reserved

                    # STEP 5: Clear cart
                    logger.info(f"Step 5: Clearing cart")
                    self.cart_service.clear_cart(user_id)

                    # SUCCESS: Publish event
                    self.kafka.send(
                        'order_events',
                        {
                            'event_type': 'order_created',
                            'order_id': order_id,
                            'user_id': user_id,
                            'total': float(cart['summary']['total']),
                            'items': cart['items']
                        }
                    )

                    logger.info(f"Order creation complete: {order_id}")

                    return {
                        'success': True,
                        'order_id': order_id,
                        'status': 'processing',
                        'total': cart['summary']['total']
                    }

                except InventoryReservationError as e:
                    # COMPENSATION: Release any successful reservations
                    logger.error(f"Inventory reservation failed: {e}")

                    for reservation_id in reservations:
                        self.inventory.release_reservation(reservation_id)

                    return {
                        'success': False,
                        'error': 'out_of_stock',
                        'message': str(e)
                    }

                except PaymentFailedError as e:
                    # COMPENSATION: Release inventory reservations
                    logger.error(f"Payment failed: {e}")

                    for reservation_id in reservations:
                        self.inventory.release_reservation(reservation_id)

                    return {
                        'success': False,
                        'error': 'payment_failed',
                        'message': str(e)
                    }

                except Exception as e:
                    # COMPENSATION: Rollback everything
                    logger.exception(f"Order creation failed: {e}")

                    # Release inventory
                    for reservation_id in reservations:
                        self.inventory.release_reservation(reservation_id)

                    # Refund payment if charged
                    if payment_transaction_id:
                        self.payment.refund_payment(
                            payment_transaction_id,
                            reason='Order creation failed'
                        )

                    # Mark order as failed if created
                    if order_id:
                        with Session(self.db) as session:
                            order = session.get(Order, order_id)
                            order.status = 'failed'
                            session.commit()

                    return {
                        'success': False,
                        'error': 'order_creation_failed',
                        'message': 'An error occurred. Your payment has been refunded.'
                    }
        ```

        ---

        ## Idempotency Key Generation

        **Client-side:**

        ```javascript
        // Generate idempotency key on checkout button click
        function checkout() {
            const idempotencyKey = `order_${Date.now()}_${userId}_${uuidv4()}`;

            // Store in session storage (survives page refresh)
            sessionStorage.setItem('checkout_idempotency_key', idempotencyKey);

            // Submit order
            fetch('/api/v1/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Idempotency-Key': idempotencyKey
                },
                body: JSON.stringify({
                    cart_id: cartId,
                    shipping_address: shippingAddress,
                    payment_method: paymentMethod
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear idempotency key on success
                    sessionStorage.removeItem('checkout_idempotency_key');
                    window.location.href = `/orders/${data.order_id}`;
                } else {
                    // Keep idempotency key for retry
                    alert(data.message);
                }
            })
            .catch(error => {
                // Network error - can safely retry with same key
                console.error('Checkout failed:', error);
                alert('Network error. Please try again.');
            });
        }
        ```

    === "⚡ Flash Sales"

        ## The Challenge

        **Problem:** Flash sale for 1,000 items. 100,000 users click "Buy Now" simultaneously. System must handle 10x normal traffic without crashing or overselling.

        **Requirements:**
        - **Fairness:** First-come-first-served (no bots advantage)
        - **No overselling:** Exactly 1,000 orders, not 1,001
        - **Graceful degradation:** Other site functions remain available
        - **Fast feedback:** Users know immediately if they got item

        ---

        ## Solution: Queue-Based Flash Sale

        **Architecture:**

        1. **Pre-queue:** Users wait in virtual queue before sale starts
        2. **Rate limiting:** Limit checkout requests to sustainable rate
        3. **Inventory pooling:** Reserve flash sale inventory separately
        4. **Circuit breaker:** Protect downstream services from overload

        ---

        ## Implementation

        ```python
        import redis
        import time
        from datetime import datetime, timedelta

        class FlashSaleService:
            """
            Flash sale management with queue-based fairness

            Handles high concurrency without overselling
            """

            def __init__(self, redis_client, inventory_service):
                self.redis = redis_client
                self.inventory = inventory_service

            def create_flash_sale(
                self,
                sale_id: str,
                product_id: int,
                quantity: int,
                sale_price: Decimal,
                start_time: datetime,
                end_time: datetime
            ):
                """
                Create flash sale

                Reserves inventory for flash sale

                Args:
                    sale_id: Unique sale identifier
                    product_id: Product on sale
                    quantity: Quantity available
                    sale_price: Discounted price
                    start_time: Sale start time
                    end_time: Sale end time
                """
                # Reserve inventory for flash sale
                reservation = self.inventory.reserve_inventory(
                    product_id=product_id,
                    quantity=quantity,
                    user_id=0  # System reservation
                )

                if not reservation['success']:
                    raise ValueError(f"Cannot reserve inventory: {reservation['error']}")

                # Store flash sale details
                sale_key = f"flash_sale:{sale_id}"
                self.redis.hset(sale_key, mapping={
                    'product_id': product_id,
                    'quantity': quantity,
                    'remaining': quantity,
                    'sale_price': float(sale_price),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'reservation_id': reservation['reservation_id']
                })

                logger.info(f"Flash sale created: {sale_id}, {quantity} units")

            def join_queue(self, sale_id: str, user_id: int) -> dict:
                """
                Add user to flash sale queue

                Args:
                    sale_id: Sale identifier
                    user_id: User joining queue

                Returns:
                    Queue position
                """
                queue_key = f"flash_sale:{sale_id}:queue"

                # Check if already in queue
                score = self.redis.zscore(queue_key, user_id)
                if score:
                    position = self.redis.zrank(queue_key, user_id) + 1
                    return {
                        'success': True,
                        'position': position,
                        'already_queued': True
                    }

                # Add to queue with timestamp as score (FIFO)
                timestamp = time.time()
                self.redis.zadd(queue_key, {user_id: timestamp})

                # Get position
                position = self.redis.zrank(queue_key, user_id) + 1

                logger.info(f"User {user_id} joined queue for {sale_id}, position: {position}")

                return {
                    'success': True,
                    'position': position,
                    'already_queued': False
                }

            def process_queue(self, sale_id: str, batch_size: int = 100):
                """
                Process queue in batches

                Background worker: Processes queue at controlled rate

                Args:
                    sale_id: Sale identifier
                    batch_size: Users to process per batch
                """
                sale_key = f"flash_sale:{sale_id}"
                queue_key = f"flash_sale:{sale_id}:queue"
                processing_key = f"flash_sale:{sale_id}:processing"

                while True:
                    # Check remaining inventory
                    remaining = int(self.redis.hget(sale_key, 'remaining') or 0)
                    if remaining <= 0:
                        logger.info(f"Flash sale {sale_id} sold out")
                        break

                    # Get next batch from queue
                    users = self.redis.zrange(queue_key, 0, batch_size - 1)
                    if not users:
                        logger.info(f"Queue empty for {sale_id}")
                        break

                    # Process each user
                    for user_id in users:
                        user_id = int(user_id)

                        # Move to processing set
                        self.redis.sadd(processing_key, user_id)
                        self.redis.zrem(queue_key, user_id)

                        # Grant purchase permission (5 minute TTL)
                        permission_key = f"flash_sale:{sale_id}:permission:{user_id}"
                        self.redis.setex(permission_key, 300, '1')

                        # Notify user (via WebSocket or push notification)
                        self._notify_user(user_id, sale_id)

                        logger.info(f"Granted purchase permission: {sale_id}, user {user_id}")

                    # Rate limiting: Wait before next batch
                    time.sleep(1)  # Process 100 users per second

            def purchase_flash_sale_item(
                self,
                sale_id: str,
                user_id: int,
                quantity: int,
                payment_method_id: str
            ) -> dict:
                """
                Purchase flash sale item

                User must have permission from queue processing

                Args:
                    sale_id: Sale identifier
                    user_id: User purchasing
                    quantity: Quantity to purchase
                    payment_method_id: Payment method

                Returns:
                    Purchase result
                """
                # Check permission
                permission_key = f"flash_sale:{sale_id}:permission:{user_id}"
                has_permission = self.redis.exists(permission_key)

                if not has_permission:
                    return {
                        'success': False,
                        'error': 'no_permission',
                        'message': 'You must wait in queue first'
                    }

                # Use Lua script for atomic inventory check and decrement
                lua_script = """
                local sale_key = KEYS[1]
                local quantity = tonumber(ARGV[1])

                local remaining = tonumber(redis.call('HGET', sale_key, 'remaining'))

                if remaining >= quantity then
                    redis.call('HINCRBY', sale_key, 'remaining', -quantity)
                    return remaining - quantity
                else
                    return -1
                end
                """

                sale_key = f"flash_sale:{sale_id}"
                new_remaining = self.redis.eval(lua_script, 1, sale_key, quantity)

                if new_remaining < 0:
                    # Sold out
                    return {
                        'success': False,
                        'error': 'sold_out',
                        'message': 'Sorry, this item is now sold out'
                    }

                # Get sale details
                sale_data = self.redis.hgetall(sale_key)
                product_id = int(sale_data[b'product_id'])
                sale_price = float(sale_data[b'sale_price'])

                # Create order (simplified)
                cart = {
                    'items': [{
                        'product_id': product_id,
                        'quantity': quantity,
                        'unit_price': sale_price,
                        'subtotal': sale_price * quantity
                    }],
                    'summary': {
                        'subtotal': sale_price * quantity,
                        'tax': sale_price * quantity * 0.09,
                        'shipping': 0.0,
                        'total': sale_price * quantity * 1.09
                    }
                }

                # Process order
                order_result = self.order_service.create_order(
                    user_id=user_id,
                    cart=cart,
                    shipping_address={},  # Pre-filled
                    payment_method_id=payment_method_id,
                    idempotency_key=f"flash_sale_{sale_id}_{user_id}_{int(time.time())}"
                )

                if order_result['success']:
                    # Remove permission
                    self.redis.delete(permission_key)

                    # Remove from processing set
                    processing_key = f"flash_sale:{sale_id}:processing"
                    self.redis.srem(processing_key, user_id)

                    logger.info(f"Flash sale purchase successful: {sale_id}, user {user_id}")

                    return {
                        'success': True,
                        'order_id': order_result['order_id']
                    }
                else:
                    # Payment failed - return inventory
                    self.redis.hincrby(sale_key, 'remaining', quantity)

                    return {
                        'success': False,
                        'error': 'payment_failed',
                        'message': order_result.get('message', 'Payment failed')
                    }

            def _notify_user(self, user_id: int, sale_id: str):
                """
                Notify user they can purchase

                Send via WebSocket or push notification

                Args:
                    user_id: User to notify
                    sale_id: Sale identifier
                """
                # Publish to user's WebSocket channel
                self.redis.publish(
                    f"user:{user_id}:notifications",
                    json.dumps({
                        'type': 'flash_sale_ready',
                        'sale_id': sale_id,
                        'message': 'You can now purchase! Hurry, 5 minutes remaining.',
                        'expires_at': (datetime.utcnow() + timedelta(minutes=5)).isoformat()
                    })
                )
        ```

        ---

        ## Rate Limiting & Circuit Breaker

        **Protect services during traffic spikes:**

        ```python
        from functools import wraps
        import time

        class RateLimiter:
            """
            Token bucket rate limiter

            Limits requests per user and globally
            """

            def __init__(self, redis_client):
                self.redis = redis_client

            def check_rate_limit(
                self,
                key: str,
                limit: int,
                window: int
            ) -> bool:
                """
                Check if rate limit exceeded

                Args:
                    key: Rate limit key (e.g., "user:123" or "global")
                    limit: Max requests in window
                    window: Time window in seconds

                Returns:
                    True if allowed, False if rate limited
                """
                current = time.time()
                window_start = current - window

                # Remove old entries
                self.redis.zremrangebyscore(key, 0, window_start)

                # Count requests in window
                count = self.redis.zcard(key)

                if count >= limit:
                    return False

                # Add current request
                self.redis.zadd(key, {current: current})
                self.redis.expire(key, window)

                return True


        def rate_limit(limit: int, window: int):
            """
            Rate limit decorator

            Args:
                limit: Max requests
                window: Time window in seconds
            """
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # Extract user_id from args/kwargs
                    user_id = kwargs.get('user_id') or args[0]

                    limiter = RateLimiter(redis_client)

                    # Per-user rate limit
                    user_key = f"rate_limit:user:{user_id}"
                    if not limiter.check_rate_limit(user_key, limit, window):
                        return {
                            'success': False,
                            'error': 'rate_limit_exceeded',
                            'message': 'Too many requests. Please try again later.'
                        }

                    # Global rate limit (protect backend)
                    global_key = f"rate_limit:global:{func.__name__}"
                    if not limiter.check_rate_limit(global_key, limit * 1000, window):
                        return {
                            'success': False,
                            'error': 'service_overloaded',
                            'message': 'Service is experiencing high traffic. Please try again.'
                        }

                    return func(*args, **kwargs)

                return wrapper
            return decorator


        # Usage
        @rate_limit(limit=10, window=60)  # 10 requests per minute
        def add_to_cart(user_id: int, product_id: int, quantity: int):
            # Implementation
            pass
        ```

        ---

        ## Flash Sale Best Practices

        | Practice | Why | Implementation |
        |----------|-----|----------------|
        | **Pre-queue** | Manage expectations, prevent stampede | Virtual queue before sale starts |
        | **Rate limiting** | Protect backend from overload | Token bucket, per-user + global |
        | **Inventory pooling** | Isolate flash sale from regular inventory | Separate reservation pool |
        | **Permission tokens** | Fair allocation, prevent bots | Time-limited purchase tokens |
        | **Circuit breaker** | Graceful degradation on overload | Fail fast, fallback responses |
        | **CDN caching** | Reduce origin load | Cache product pages, images |

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling e-commerce from 1M to 300M customers.

    **Scaling challenges at 300M DAU:**

    - **Read throughput:** 97K read QPS (product browsing, search)
    - **Write throughput:** 11K write QPS (orders, cart updates, inventory)
    - **Storage:** 543 TB of data
    - **Inventory consistency:** Prevent overselling at scale
    - **Flash sales:** 10x traffic spikes

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Product DB reads** | ✅ Yes | Read replicas (10 replicas), Redis cache (5 min TTL) |
    | **Inventory DB writes** | ✅ Yes | Connection pooling, optimistic locking for low contention |
    | **Cart Redis** | 🟡 Approaching | Redis Cluster (20 nodes), replication for HA |
    | **Elasticsearch** | ✅ Yes | 50 shards by category, 3x replication, hot/cold tiers |
    | **Order DB writes** | 🟢 No | 23 QPS manageable, but shard for future growth |
    | **Payment gateway** | ✅ Yes | Retry logic, multiple payment providers (Stripe + PayPal) |

    ---

    ## Database Sharding Strategy

    **Orders & Users: Shard by user_id**

    ```python
    def get_shard_id(user_id: int, num_shards: int = 16) -> int:
        """
        Determine shard for user

        Consistent hashing for even distribution

        Args:
            user_id: User ID
            num_shards: Total number of shards

        Returns:
            Shard ID (0 to num_shards-1)
        """
        return user_id % num_shards


    # Database connection routing
    def get_db_connection(user_id: int):
        """Get database connection for user's shard"""
        shard_id = get_shard_id(user_id)
        return db_connections[f"shard_{shard_id}"]
    ```

    **Products: No sharding (manageable at 100M)**

    - Use read replicas for scale-out reads
    - Redis cache for hot products

    **Inventory: Shard by product_id**

    ```python
    def get_inventory_shard(product_id: int, num_shards: int = 8) -> int:
        """Shard inventory by product_id"""
        return product_id % num_shards
    ```

    ---

    ## Caching Strategy

    | Data Type | Cache | TTL | Invalidation |
    |-----------|-------|-----|--------------|
    | **Product catalog** | Redis | 5 minutes | On product update (pub/sub) |
    | **Product images** | CDN | 1 day | On image upload |
    | **Search results** | Redis | 5 minutes | On product update |
    | **Inventory counts** | Redis | 60 seconds | On inventory change |
    | **Shopping cart** | Redis | 30 days | On cart update, auto-expire |
    | **User sessions** | Redis | 24 hours | On logout, auto-expire |
    | **Price cache** | Redis | 1 hour | On price change |

    **Multi-level caching:**

    ```
    Request → CDN → Redis → Database

    CDN: Static assets (images, CSS, JS)
    Redis: Dynamic data (products, cart, inventory)
    Database: Source of truth
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 300M DAU:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $129,600 (900 × m5.2xlarge) |
    | **RDS PostgreSQL (Products)** | $43,200 (1 master + 10 replicas) |
    | **RDS PostgreSQL (Orders)** | $86,400 (16 shards) |
    | **RDS PostgreSQL (Inventory)** | $43,200 (8 shards) |
    | **Redis Cluster** | $64,800 (30 nodes) |
    | **Elasticsearch** | $108,000 (50 nodes) |
    | **S3 storage** | $12,900 (543 TB) |
    | **CDN** | $212,500 (2,500 TB egress) |
    | **Kafka** | $21,600 (20 brokers) |
    | **Payment gateway fees** | $300,000 (2M orders × 2.9% × $50) |
    | **Total** | **$1,022,200/month** |

    **Optimization strategies:**

    - **Reserved instances:** 40% savings on EC2/RDS
    - **Spot instances:** Non-critical workloads (analytics)
    - **S3 Intelligent Tiering:** 30% savings on old images
    - **CDN optimization:** Compress images, lazy loading
    - **Database optimization:** Indexes, query tuning

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Product Page Latency (P95)** | < 200ms | > 500ms |
    | **Checkout Latency (P95)** | < 2s | > 5s |
    | **Search Latency (P95)** | < 100ms | > 300ms |
    | **Order Success Rate** | > 99% | < 95% |
    | **Payment Success Rate** | > 97% | < 90% |
    | **Inventory Accuracy** | 100% | < 99.9% |
    | **Cart Cache Hit Rate** | > 90% | < 80% |
    | **Database Connection Pool** | < 80% | > 90% |

    ---

    ## Performance Optimization

    **Database query optimization:**

    ```sql
    -- Bad: N+1 query
    SELECT * FROM orders WHERE user_id = 123;
    -- Then for each order:
    SELECT * FROM order_items WHERE order_id = ?;

    -- Good: JOIN with batch fetch
    SELECT o.*, oi.*
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.user_id = 123
    ORDER BY o.created_at DESC
    LIMIT 20;

    -- Add index
    CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);
    ```

    **API response optimization:**

    ```python
    # Bad: Return full product objects
    @app.get("/api/v1/search")
    def search(query: str):
        products = es.search(query)
        # Fetch full details for 100 products (slow)
        return [get_product_details(p.id) for p in products]

    # Good: Return minimal data, lazy load details
    @app.get("/api/v1/search")
    def search(query: str):
        products = es.search(query)
        # Return IDs and cached data only
        return [
            {
                'product_id': p.id,
                'name': p.name,
                'price': p.price,
                'image': p.thumbnail,  # Cached URL
                'rating': p.rating
            }
            for p in products
        ]
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Strong consistency for inventory:** Pessimistic locking prevents overselling
    2. **Eventual consistency for catalog:** Product updates can have brief delay
    3. **Redis for carts:** Fast, persistent, TTL-based expiry
    4. **Idempotent payments:** Prevent duplicate charges on retry
    5. **Database sharding:** Orders and users sharded by user_id
    6. **Queue-based flash sales:** Fair allocation during traffic spikes
    7. **Multi-level caching:** CDN → Redis → Database

    ---

    ## Interview Tips

    ✅ **Emphasize inventory consistency** - Preventing overselling is critical

    ✅ **Discuss payment idempotency** - Duplicate charges = bad customer experience

    ✅ **Explain ACID transactions** - Order creation must be atomic

    ✅ **Address flash sales** - 10x traffic spikes are common

    ✅ **Cost awareness** - Payment gateway fees are significant

    ✅ **Security considerations** - PCI compliance, never store raw card data

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent overselling?"** | Pessimistic locking (SELECT FOR UPDATE), inventory reservation system |
    | **"How to handle payment failures?"** | Idempotency keys, retry logic, compensation transactions (refund) |
    | **"How to scale for flash sales?"** | Queue-based system, rate limiting, inventory pooling, circuit breaker |
    | **"How to handle distributed transactions?"** | Saga pattern: reserve → pay → create → consume, with compensations |
    | **"How to ensure cart persistence?"** | Redis for hot carts, PostgreSQL for cold carts, lazy loading |
    | **"How to handle high read traffic?"** | Read replicas, multi-level caching (CDN + Redis), Elasticsearch |

    ---

    ## Extended Topics (If Time Permits)

    **Product recommendations:**
    - Collaborative filtering (users who bought X also bought Y)
    - Content-based filtering (similar products)
    - Real-time personalization (browsing history)

    **Fraud detection:**
    - Velocity checks (too many orders in short time)
    - Device fingerprinting
    - Address verification
    - Behavioral analysis

    **Order fulfillment:**
    - Warehouse management integration
    - Shipping label generation
    - Tracking updates
    - Return processing

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Amazon, eBay, Shopify, Walmart, Alibaba, Target, Etsy

---

*Master this problem and you'll be ready for: Payment systems, booking platforms, ticketing systems, marketplace platforms*
