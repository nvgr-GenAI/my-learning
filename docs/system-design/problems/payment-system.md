# Design Payment System (Stripe)

A global payment processing platform that enables businesses to accept online payments, manage subscriptions, detect fraud, and handle refunds with strong consistency guarantees.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M transactions/day, $50B payment volume/year, 99.99% availability |
| **Key Challenges** | ACID transactions, idempotency, fraud detection, PCI compliance, exactly-once processing |
| **Core Concepts** | Double-entry ledger, idempotency keys, 2PC, reconciliation, chargebacks |
| **Companies** | Stripe, PayPal, Square, Adyen, Braintree, Razorpay |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Process Payment** | Accept payments via credit/debit cards | P0 (Must have) |
    | **Refunds** | Full and partial refunds | P0 (Must have) |
    | **Recurring Billing** | Subscription payments, auto-retry | P0 (Must have) |
    | **Webhooks** | Event notifications to merchants | P0 (Must have) |
    | **Fraud Detection** | Real-time fraud scoring | P0 (Must have) |
    | **Multi-currency** | Support 135+ currencies | P1 (Should have) |
    | **Payouts** | Batch payouts to sellers/vendors | P1 (Should have) |
    | **Dispute Management** | Handle chargebacks | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Buy Now Pay Later (BNPL)
    - Crypto payments
    - Point-of-sale (POS) terminals
    - Banking-as-a-Service
    - Tax calculation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Consistency** | Strong consistency | Money must be accurate, no lost transactions |
    | **Availability** | 99.99% uptime | 52 minutes downtime/year acceptable |
    | **Latency** | < 2s for payment processing | User expectation, faster = higher conversion |
    | **Idempotency** | Exactly-once processing | Prevent duplicate charges on retry |
    | **Durability** | 99.999999999% | Financial data cannot be lost |
    | **Compliance** | PCI DSS Level 1 | Required for card processing |
    | **Auditability** | Full transaction history | Required for compliance, disputes |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Transactions: 100M
    Payment QPS: 100M / 86,400 = ~1,160 TPS
    Peak QPS: 3x average = ~3,500 TPS (Black Friday, holidays)

    Refunds:
    - Refund rate: ~5% of transactions
    - Daily refunds: 5M
    - Refund QPS: 5M / 86,400 = ~58 TPS

    Webhooks:
    - 3 events per transaction avg (created, succeeded, captured)
    - Daily webhook events: 100M × 3 = 300M
    - Webhook QPS: 300M / 86,400 = ~3,470 events/sec

    Fraud checks:
    - All transactions checked
    - Fraud QPS: ~1,160 checks/sec

    Read operations:
    - Transaction queries: 10x writes = 11,600 QPS
    - Dashboard queries: 2,000 QPS

    Total Read QPS: ~13,600
    Total Write QPS: ~5,000 (payments + refunds + ledger entries)
    Read/Write ratio: 2.7:1
    ```

    ### Storage Estimates

    ```
    Transaction storage:
    - Transaction record: 2 KB (payment details, metadata)
    - Card token: 1 KB (encrypted)
    - Ledger entries: 2 entries × 1 KB = 2 KB (double-entry)
    - Total per transaction: ~5 KB

    For 10 years:
    - Transactions: 100M/day × 365 × 10 = 365 billion transactions
    - Storage: 365B × 5 KB = 1.825 PB

    Webhook logs:
    - 300M events/day × 5 KB = 1.5 TB/day
    - 10 years: 1.5 TB × 365 × 10 = 5.475 PB

    Fraud model data:
    - Historical data for ML: 500 GB
    - Real-time features: 100 GB

    User/merchant data:
    - 10M merchants × 100 KB = 1 TB
    - 500M customers × 20 KB = 10 TB

    Total: 1.825 PB (transactions) + 5.475 PB (webhooks) + 11 TB (users) ≈ 7.3 PB
    ```

    ### Bandwidth Estimates

    ```
    Payment ingress:
    - 1,160 TPS × 5 KB = 5.8 MB/sec ≈ 46 Mbps

    Webhook egress:
    - 3,470 events/sec × 5 KB = 17.35 MB/sec ≈ 139 Mbps

    Dashboard queries:
    - 13,600 QPS × 10 KB = 136 MB/sec ≈ 1.09 Gbps

    Total ingress: ~50 Mbps
    Total egress: ~1.2 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Hot transactions (last 24 hours):
    - Transactions: 100M × 5 KB = 500 GB
    - Cache 20% hottest: 100 GB

    Merchant sessions:
    - 1M concurrent merchants × 50 KB = 50 GB

    Fraud model cache:
    - ML model: 10 GB
    - Real-time features: 100 GB

    Idempotency keys:
    - 24-hour window: 100M × 100 bytes = 10 GB

    Rate limiting:
    - Token buckets for 10M merchants: 1 GB

    Total cache: 100 GB + 50 GB + 110 GB + 10 GB + 1 GB ≈ 270 GB
    ```

    ---

    ## Key Assumptions

    1. Average transaction value: $50
    2. Card payments: 70%, ACH: 20%, wallets: 10%
    3. Strong consistency required (ACID transactions)
    4. Exactly-once processing (idempotency critical)
    5. Multi-region deployment for disaster recovery
    6. Regulatory compliance: PCI DSS, PSD2, SOC 2

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Strong consistency:** ACID transactions for payment processing
    2. **Idempotency:** Prevent duplicate charges using idempotency keys
    3. **Event-driven:** Async processing for webhooks, fraud, reconciliation
    4. **Defense in depth:** Multiple security layers (encryption, tokenization, fraud)
    5. **Auditability:** Immutable audit log for compliance

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Merchant[Merchant App]
            Mobile[Mobile SDK]
            Web[Web Checkout]
        end

        subgraph "Edge Layer"
            LB[Load Balancer<br/>TLS termination]
            WAF[WAF<br/>DDoS protection]
        end

        subgraph "API Layer"
            Payment_API[Payment Service<br/>Process payments]
            Refund_API[Refund Service<br/>Process refunds]
            Subscription_API[Subscription Service<br/>Recurring billing]
            Webhook_API[Webhook Service<br/>Event delivery]
            Dashboard_API[Dashboard Service<br/>Analytics, reports]
        end

        subgraph "Core Processing"
            Idempotency[Idempotency Service<br/>Deduplicate requests]
            Fraud[Fraud Detection<br/>Risk scoring]
            Ledger[Ledger Service<br/>Double-entry accounting]
            Card_Processor[Card Processor<br/>Authorize, capture]
        end

        subgraph "External"
            Visa[Visa/Mastercard<br/>Card networks]
            ACH[ACH Network<br/>Bank transfers]
            Bank[Issuing Bank<br/>Card verification]
        end

        subgraph "Data Processing"
            Event_Bus[Kafka<br/>Event streaming]
            Webhook_Worker[Webhook Worker<br/>Deliver events]
            Recon_Worker[Reconciliation Worker<br/>Match settlements]
            Fraud_ML[Fraud ML Pipeline<br/>Model training]
            Subscription_Worker[Subscription Worker<br/>Auto-charge]
        end

        subgraph "Caching"
            Redis_Idempotency[Redis<br/>Idempotency keys]
            Redis_Session[Redis<br/>Sessions]
            Redis_Fraud[Redis<br/>Fraud features]
            Redis_Rate[Redis<br/>Rate limiting]
        end

        subgraph "Storage"
            Payment_DB[(Payment DB<br/>PostgreSQL<br/>Transactions)]
            Ledger_DB[(Ledger DB<br/>PostgreSQL<br/>Accounting)]
            Merchant_DB[(Merchant DB<br/>PostgreSQL<br/>Merchant data)]
            Audit_DB[(Audit Log<br/>Cassandra<br/>Immutable)]
            Fraud_DB[(Fraud DB<br/>Time-series<br/>Events)]
        end

        Merchant --> WAF
        Mobile --> WAF
        Web --> WAF

        WAF --> LB
        LB --> Payment_API
        LB --> Refund_API
        LB --> Subscription_API
        LB --> Dashboard_API

        Payment_API --> Idempotency
        Idempotency --> Redis_Idempotency
        Idempotency --> Fraud
        Fraud --> Redis_Fraud
        Fraud --> Fraud_DB
        Fraud --> Card_Processor

        Card_Processor --> Visa
        Card_Processor --> ACH
        Card_Processor --> Bank

        Payment_API --> Ledger
        Ledger --> Ledger_DB
        Payment_API --> Payment_DB
        Payment_API --> Audit_DB

        Payment_API --> Event_Bus
        Refund_API --> Event_Bus
        Subscription_API --> Event_Bus

        Event_Bus --> Webhook_Worker
        Event_Bus --> Recon_Worker
        Event_Bus --> Fraud_ML
        Event_Bus --> Subscription_Worker

        Webhook_Worker --> Webhook_API
        Webhook_API --> Merchant

        Subscription_Worker --> Payment_API

        Dashboard_API --> Payment_DB
        Dashboard_API --> Ledger_DB

        style WAF fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Idempotency fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Redis_Fraud fill:#fff4e1
        style Redis_Rate fill:#fff4e1
        style Payment_DB fill:#ffe1e1
        style Ledger_DB fill:#ffe1e1
        style Merchant_DB fill:#ffe1e1
        style Audit_DB fill:#ffe1e1
        style Fraud_DB fill:#e1f5e1
        style Event_Bus fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PostgreSQL (Payment DB)** | ACID guarantees, foreign keys, strong consistency | MySQL (similar), MongoDB (no ACID), DynamoDB (eventual consistency) |
    | **Redis (Idempotency)** | Fast key-value lookup (<1ms), TTL support | PostgreSQL (too slow), Memcached (no persistence) |
    | **Kafka** | Event streaming, exactly-once delivery, replay capability | RabbitMQ (lower throughput), SQS (eventual delivery) |
    | **Ledger Service** | Double-entry accounting, financial audit trail | Custom SQL (error-prone), manual reconciliation |
    | **Fraud ML** | Real-time risk scoring, adaptive learning | Rule-based (static, easy to bypass), manual review (slow) |
    | **Cassandra (Audit)** | Immutable append-only log, time-series optimized | PostgreSQL (scaling limits), S3 (no real-time queries) |

    **Key Trade-off:** We chose **strong consistency over availability**. Payments require accuracy; we accept brief unavailability during failures rather than risk duplicate charges or lost money.

    ---

    ## API Design

    ### 1. Create Payment Intent

    **Request:**
    ```http
    POST /api/v1/payment_intents
    Content-Type: application/json
    Authorization: Bearer sk_live_xyz123
    Idempotency-Key: unique_request_id_123

    {
      "amount": 5000,                     // Amount in cents
      "currency": "usd",
      "payment_method": "pm_card_visa",
      "customer": "cus_123456",
      "description": "Order #1234",
      "metadata": {
        "order_id": "1234",
        "user_id": "user_789"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "pi_1234567890",
      "object": "payment_intent",
      "amount": 5000,
      "currency": "usd",
      "status": "requires_confirmation",
      "client_secret": "pi_1234567890_secret_xyz",
      "payment_method": "pm_card_visa",
      "customer": "cus_123456",
      "created": 1706553600,
      "next_action": {
        "type": "redirect_to_url",
        "redirect_to_url": {
          "url": "https://3ds.stripe.com/authenticate/xyz"
        }
      }
    }
    ```

    **Design Notes:**

    - Idempotency-Key header prevents duplicate charges on retry
    - Return immediately with intent status (async processing)
    - Client secret for secure client-side confirmation
    - Support 3D Secure for strong authentication (PSD2 compliance)

    ---

    ### 2. Confirm Payment

    **Request:**
    ```http
    POST /api/v1/payment_intents/pi_1234567890/confirm
    Content-Type: application/json
    Authorization: Bearer sk_live_xyz123
    Idempotency-Key: unique_confirm_id_456

    {
      "payment_method": "pm_card_visa",
      "return_url": "https://merchant.com/return"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "pi_1234567890",
      "object": "payment_intent",
      "amount": 5000,
      "status": "succeeded",
      "charges": {
        "data": [
          {
            "id": "ch_1234567890",
            "amount": 5000,
            "captured": true,
            "payment_method_details": {
              "card": {
                "brand": "visa",
                "last4": "4242",
                "exp_month": 12,
                "exp_year": 2025
              }
            }
          }
        ]
      }
    }
    ```

    **Design Notes:**

    - Synchronous confirmation (wait for card network response)
    - Return full charge object with payment details
    - Trigger webhook events asynchronously

    ---

    ### 3. Create Refund

    **Request:**
    ```http
    POST /api/v1/refunds
    Content-Type: application/json
    Authorization: Bearer sk_live_xyz123
    Idempotency-Key: unique_refund_id_789

    {
      "charge": "ch_1234567890",
      "amount": 2000,                    // Partial refund
      "reason": "requested_by_customer",
      "metadata": {
        "reason_detail": "Product damaged"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "re_1234567890",
      "object": "refund",
      "amount": 2000,
      "charge": "ch_1234567890",
      "currency": "usd",
      "status": "succeeded",
      "reason": "requested_by_customer",
      "created": 1706553700
    }
    ```

    **Design Notes:**

    - Support partial and full refunds
    - Idempotent (same key returns same refund)
    - Update ledger with refund entries
    - Trigger refund.succeeded webhook

    ---

    ### 4. Create Subscription

    **Request:**
    ```http
    POST /api/v1/subscriptions
    Content-Type: application/json
    Authorization: Bearer sk_live_xyz123

    {
      "customer": "cus_123456",
      "items": [
        {
          "price": "price_monthly_99",
          "quantity": 1
        }
      ],
      "payment_method": "pm_card_visa",
      "default_payment_method": "pm_card_visa"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "sub_1234567890",
      "object": "subscription",
      "status": "active",
      "customer": "cus_123456",
      "current_period_start": 1706553600,
      "current_period_end": 1709232000,
      "items": {
        "data": [
          {
            "id": "si_1234567890",
            "price": {
              "id": "price_monthly_99",
              "unit_amount": 9900,
              "currency": "usd",
              "recurring": {
                "interval": "month",
                "interval_count": 1
              }
            }
          }
        ]
      },
      "latest_invoice": "in_1234567890"
    }
    ```

    **Design Notes:**

    - Automatically charge customer every billing period
    - Smart retry logic for failed payments
    - Proration support for mid-cycle changes
    - Cancel, pause, or update subscriptions

    ---

    ## Database Schema

    ### Payments (PostgreSQL)

    ```sql
    -- Payment intents
    CREATE TABLE payment_intents (
        id VARCHAR(64) PRIMARY KEY,
        merchant_id VARCHAR(64) NOT NULL,
        customer_id VARCHAR(64),
        amount BIGINT NOT NULL,                    -- Amount in cents
        currency VARCHAR(3) NOT NULL,
        status VARCHAR(32) NOT NULL,               -- requires_confirmation, processing, succeeded, failed
        payment_method_id VARCHAR(64),
        idempotency_key VARCHAR(64) UNIQUE,
        client_secret VARCHAR(128),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_merchant_created (merchant_id, created_at DESC),
        INDEX idx_customer (customer_id),
        INDEX idx_idempotency (idempotency_key),
        INDEX idx_status (status)
    );

    -- Charges (actual payment attempts)
    CREATE TABLE charges (
        id VARCHAR(64) PRIMARY KEY,
        payment_intent_id VARCHAR(64) NOT NULL,
        amount BIGINT NOT NULL,
        currency VARCHAR(3) NOT NULL,
        status VARCHAR(32) NOT NULL,               -- pending, succeeded, failed
        captured BOOLEAN DEFAULT FALSE,
        payment_method VARCHAR(64),
        failure_code VARCHAR(64),
        failure_message TEXT,
        network_transaction_id VARCHAR(64),        -- Card network reference
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (payment_intent_id) REFERENCES payment_intents(id),
        INDEX idx_payment_intent (payment_intent_id),
        INDEX idx_network_txn (network_transaction_id)
    );

    -- Refunds
    CREATE TABLE refunds (
        id VARCHAR(64) PRIMARY KEY,
        charge_id VARCHAR(64) NOT NULL,
        amount BIGINT NOT NULL,
        currency VARCHAR(3) NOT NULL,
        status VARCHAR(32) NOT NULL,               -- pending, succeeded, failed
        reason VARCHAR(64),
        idempotency_key VARCHAR(64) UNIQUE,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (charge_id) REFERENCES charges(id),
        INDEX idx_charge (charge_id),
        INDEX idx_idempotency (idempotency_key)
    );

    -- Payment methods (tokenized)
    CREATE TABLE payment_methods (
        id VARCHAR(64) PRIMARY KEY,
        customer_id VARCHAR(64) NOT NULL,
        type VARCHAR(32) NOT NULL,                 -- card, bank_account
        card_token VARCHAR(128),                   -- Encrypted card token
        card_last4 VARCHAR(4),
        card_brand VARCHAR(32),
        card_exp_month INT,
        card_exp_year INT,
        billing_details JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_customer (customer_id)
    );
    ```

    **Why PostgreSQL:**

    - **ACID transactions:** Critical for payment accuracy
    - **Foreign keys:** Data integrity (refund must reference charge)
    - **Strong consistency:** Read your own writes
    - **Mature tooling:** Backups, replication, monitoring

    ---

    ### Ledger (PostgreSQL)

    ```sql
    -- Double-entry ledger (immutable)
    CREATE TABLE ledger_entries (
        id BIGSERIAL PRIMARY KEY,
        transaction_id VARCHAR(64) NOT NULL,       -- Links to payment/refund
        account_id VARCHAR(64) NOT NULL,           -- Account being debited/credited
        account_type VARCHAR(32) NOT NULL,         -- merchant, customer, stripe, card_network
        debit_amount BIGINT DEFAULT 0,             -- Amount debited (positive)
        credit_amount BIGINT DEFAULT 0,            -- Amount credited (positive)
        currency VARCHAR(3) NOT NULL,
        balance_after BIGINT NOT NULL,             -- Running balance
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_transaction (transaction_id),
        INDEX idx_account_created (account_id, created_at DESC),
        CONSTRAINT debit_xor_credit CHECK (
            (debit_amount > 0 AND credit_amount = 0) OR
            (debit_amount = 0 AND credit_amount > 0)
        )
    );

    -- Accounts (for ledger)
    CREATE TABLE accounts (
        id VARCHAR(64) PRIMARY KEY,
        owner_id VARCHAR(64) NOT NULL,             -- Merchant or customer ID
        account_type VARCHAR(32) NOT NULL,         -- merchant, customer, stripe
        currency VARCHAR(3) NOT NULL,
        balance BIGINT DEFAULT 0,                  -- Current balance in cents
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_owner (owner_id)
    );
    ```

    **Double-Entry Accounting Example:**

    For a $50 payment with $1.50 fee:

    | Account | Debit | Credit | Description |
    |---------|-------|--------|-------------|
    | Customer | $50 | - | Customer pays |
    | Stripe | - | $50 | Stripe receives |
    | Stripe | $48.50 | - | Stripe pays merchant |
    | Merchant | - | $48.50 | Merchant receives |
    | Stripe | $1.50 | - | Stripe fee |
    | Stripe Revenue | - | $1.50 | Stripe revenue |

    **Benefits:**

    - **Balanced books:** Total debits = total credits (always)
    - **Audit trail:** Every transaction has two sides
    - **Reconciliation:** Easy to match with bank statements

    ---

    ### Subscriptions (PostgreSQL)

    ```sql
    -- Subscriptions
    CREATE TABLE subscriptions (
        id VARCHAR(64) PRIMARY KEY,
        customer_id VARCHAR(64) NOT NULL,
        merchant_id VARCHAR(64) NOT NULL,
        status VARCHAR(32) NOT NULL,               -- active, past_due, canceled
        current_period_start TIMESTAMP NOT NULL,
        current_period_end TIMESTAMP NOT NULL,
        billing_cycle_anchor TIMESTAMP NOT NULL,
        default_payment_method_id VARCHAR(64),
        cancel_at_period_end BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_customer (customer_id),
        INDEX idx_merchant (merchant_id),
        INDEX idx_period_end (current_period_end)  -- For billing worker
    );

    -- Subscription items (products)
    CREATE TABLE subscription_items (
        id VARCHAR(64) PRIMARY KEY,
        subscription_id VARCHAR(64) NOT NULL,
        price_id VARCHAR(64) NOT NULL,
        quantity INT DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
        INDEX idx_subscription (subscription_id)
    );
    ```

    ---

    ### Audit Log (Cassandra)

    ```sql
    -- Immutable audit log
    CREATE TABLE audit_log (
        id TIMEUUID PRIMARY KEY,
        event_type TEXT,                           -- payment.created, payment.succeeded
        resource_type TEXT,                        -- payment_intent, charge, refund
        resource_id TEXT,
        merchant_id TEXT,
        actor_id TEXT,                             -- Who performed action
        action TEXT,                               -- create, update, delete
        changes FROZEN<MAP<TEXT, TEXT>>,           -- Before/after values
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP,
        PRIMARY KEY (merchant_id, created_at, id)
    ) WITH CLUSTERING ORDER BY (created_at DESC);
    ```

    **Why Cassandra:**

    - **Append-only:** No updates, perfect for immutable logs
    - **Time-series optimized:** Fast queries by time range
    - **High write throughput:** 5,000+ writes/sec
    - **Horizontal scaling:** Add nodes without downtime

    ---

    ## Data Flow Diagrams

    ### Payment Processing Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Payment_API
        participant Idempotency
        participant Fraud
        participant Card_Processor
        participant Ledger
        participant Payment_DB
        participant Kafka
        participant Webhook

        Client->>Payment_API: POST /payment_intents (Idempotency-Key)
        Payment_API->>Idempotency: Check idempotency key

        alt Key exists (duplicate request)
            Idempotency-->>Payment_API: Return cached response
            Payment_API-->>Client: 200 OK (idempotent)
        else New request
            Payment_API->>Fraud: Check fraud score
            Fraud-->>Payment_API: Risk score: 0.2 (low risk)

            Payment_API->>Payment_DB: BEGIN TRANSACTION
            Payment_API->>Payment_DB: INSERT payment_intent
            Payment_API->>Card_Processor: Authorize payment

            alt Card approved
                Card_Processor-->>Payment_API: Authorized (txn_id: 12345)
                Payment_API->>Payment_DB: INSERT charge (succeeded)
                Payment_API->>Ledger: Create ledger entries
                Ledger->>Ledger: Debit customer, credit merchant
                Ledger-->>Payment_API: Ledger balanced
                Payment_API->>Payment_DB: COMMIT TRANSACTION

                Payment_API->>Kafka: Publish payment.succeeded event
                Payment_API->>Idempotency: Cache response (24h TTL)
                Payment_API-->>Client: 200 OK (payment succeeded)

                Kafka->>Webhook: Process event
                Webhook->>Client: POST /webhook (payment.succeeded)
            else Card declined
                Card_Processor-->>Payment_API: Declined (insufficient_funds)
                Payment_API->>Payment_DB: INSERT charge (failed)
                Payment_API->>Payment_DB: COMMIT TRANSACTION

                Payment_API->>Kafka: Publish payment.failed event
                Payment_API-->>Client: 402 Payment Required (declined)
            end
        end
    ```

    **Flow Explanation:**

    1. **Idempotency check** - Deduplicate requests (< 5ms Redis lookup)
    2. **Fraud detection** - Real-time risk scoring (< 50ms)
    3. **Card authorization** - Contact card network (< 1s)
    4. **Ledger update** - Double-entry accounting (< 20ms)
    5. **Webhook delivery** - Async event notification (< 5s)

    ---

    ### Refund Processing Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Refund_API
        participant Payment_DB
        participant Card_Processor
        participant Ledger
        participant Kafka
        participant Webhook

        Client->>Refund_API: POST /refunds (charge_id, amount)
        Refund_API->>Payment_DB: BEGIN TRANSACTION
        Refund_API->>Payment_DB: SELECT charge (FOR UPDATE)

        alt Charge exists and refundable
            Refund_API->>Payment_DB: INSERT refund (pending)
            Refund_API->>Card_Processor: Refund transaction

            alt Refund approved
                Card_Processor-->>Refund_API: Refund accepted
                Refund_API->>Payment_DB: UPDATE refund (succeeded)
                Refund_API->>Ledger: Create refund ledger entries
                Ledger->>Ledger: Credit customer, debit merchant
                Ledger-->>Refund_API: Ledger balanced
                Refund_API->>Payment_DB: COMMIT TRANSACTION

                Refund_API->>Kafka: Publish refund.succeeded event
                Refund_API-->>Client: 200 OK (refund succeeded)

                Kafka->>Webhook: Process event
                Webhook->>Client: POST /webhook (refund.succeeded)
            else Refund failed
                Card_Processor-->>Refund_API: Refund failed
                Refund_API->>Payment_DB: UPDATE refund (failed)
                Refund_API->>Payment_DB: COMMIT TRANSACTION
                Refund_API-->>Client: 400 Bad Request (refund failed)
            end
        else Charge not found or already refunded
            Refund_API->>Payment_DB: ROLLBACK TRANSACTION
            Refund_API-->>Client: 400 Bad Request (invalid charge)
        end
    ```

    **Flow Explanation:**

    1. **Validation** - Check charge exists and refundable
    2. **Pessimistic locking** - FOR UPDATE prevents concurrent refunds
    3. **Card network refund** - Process refund with issuer
    4. **Ledger reversal** - Reverse original ledger entries
    5. **Webhook delivery** - Notify merchant of refund

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical payment system subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Idempotency** | How to prevent duplicate charges on retry? | Idempotency keys with 24-hour TTL |
    | **Ledger Design** | How to ensure accurate accounting? | Double-entry ledger with ACID transactions |
    | **Fraud Detection** | How to detect fraud in real-time? | ML-based risk scoring + rule engine |
    | **Reconciliation** | How to match payments with settlements? | Daily batch reconciliation with card networks |

    ---

    === "🔑 Idempotency"

        ## The Challenge

        **Problem:** Network failures, timeouts, or client retries can cause duplicate payment requests. A customer might be charged twice!

        **Example scenario:**

        ```
        1. Client sends payment request
        2. Server processes payment successfully
        3. Network fails before response reaches client
        4. Client retries request (thinks it failed)
        5. Without idempotency: Customer charged twice!
        ```

        ---

        ## Idempotency Keys

        **Solution:** Client provides unique `Idempotency-Key` header. Same key always returns same response.

        **Implementation:**

        ```python
        import redis
        import hashlib
        import json
        from datetime import timedelta

        class IdempotencyService:
            """Ensure exactly-once processing of payment requests"""

            KEY_TTL = timedelta(hours=24)  # Keys expire after 24 hours

            def __init__(self, redis_client):
                self.redis = redis_client

            def check_idempotency(self, idempotency_key: str, merchant_id: str) -> dict:
                """
                Check if request with this idempotency key was already processed

                Args:
                    idempotency_key: Unique request identifier from client
                    merchant_id: Merchant making request (scope keys to merchant)

                Returns:
                    Cached response if key exists, None otherwise
                """
                if not idempotency_key:
                    return None

                # Namespace key by merchant to prevent cross-merchant conflicts
                cache_key = f"idempotency:{merchant_id}:{idempotency_key}"

                # Check Redis cache
                cached_response = self.redis.get(cache_key)

                if cached_response:
                    logger.info(f"Idempotency hit for key {idempotency_key}")
                    return json.loads(cached_response)

                return None

            def store_response(
                self,
                idempotency_key: str,
                merchant_id: str,
                response: dict,
                status_code: int
            ):
                """
                Store response for idempotency key

                Args:
                    idempotency_key: Unique request identifier
                    merchant_id: Merchant ID
                    response: API response to cache
                    status_code: HTTP status code
                """
                if not idempotency_key:
                    return

                cache_key = f"idempotency:{merchant_id}:{idempotency_key}"

                # Store response with status code
                cached_data = {
                    'response': response,
                    'status_code': status_code,
                    'timestamp': datetime.utcnow().isoformat()
                }

                # Set with 24-hour expiry
                self.redis.setex(
                    cache_key,
                    self.KEY_TTL,
                    json.dumps(cached_data)
                )

                logger.info(f"Cached response for idempotency key {idempotency_key}")

            def generate_idempotency_key(self, request_data: dict) -> str:
                """
                Generate idempotency key from request data (if client doesn't provide)

                Args:
                    request_data: Request parameters

                Returns:
                    SHA-256 hash of request data
                """
                # Sort keys for consistent hashing
                canonical = json.dumps(request_data, sort_keys=True)
                return hashlib.sha256(canonical.encode()).hexdigest()
        ```

        ---

        ## Payment API with Idempotency

        ```python
        from fastapi import FastAPI, Header, HTTPException
        from typing import Optional

        app = FastAPI()

        @app.post("/api/v1/payment_intents")
        async def create_payment_intent(
            payment_data: dict,
            authorization: str = Header(...),
            idempotency_key: Optional[str] = Header(None)
        ):
            """
            Create payment intent with idempotency protection

            Args:
                payment_data: Payment details (amount, currency, etc.)
                authorization: Merchant API key
                idempotency_key: Optional client-provided idempotency key
            """
            # Extract merchant ID from API key
            merchant_id = validate_api_key(authorization)

            # Generate idempotency key if not provided
            if not idempotency_key:
                idempotency_key = idempotency_service.generate_idempotency_key(
                    {**payment_data, 'merchant_id': merchant_id}
                )

            # Check if request already processed
            cached_response = idempotency_service.check_idempotency(
                idempotency_key,
                merchant_id
            )

            if cached_response:
                # Return cached response (idempotent)
                return Response(
                    content=json.dumps(cached_response['response']),
                    status_code=cached_response['status_code']
                )

            try:
                # Process payment (first time)
                payment_intent = await process_payment_intent(payment_data, merchant_id)

                # Cache response for future retries
                idempotency_service.store_response(
                    idempotency_key,
                    merchant_id,
                    payment_intent,
                    200
                )

                return payment_intent

            except PaymentError as e:
                # Cache error response too (idempotent errors)
                error_response = {'error': str(e)}
                idempotency_service.store_response(
                    idempotency_key,
                    merchant_id,
                    error_response,
                    402
                )
                raise HTTPException(status_code=402, detail=str(e))
        ```

        ---

        ## Idempotency Key Best Practices

        | Practice | Rationale |
        |----------|-----------|
        | **Client-generated UUIDs** | Client controls uniqueness, can retry safely |
        | **24-hour expiry** | Balance between deduplication and storage cost |
        | **Scope to merchant** | Prevent cross-merchant key collisions |
        | **Cache both success and errors** | Idempotent error handling |
        | **Store status code** | Return exact same HTTP response |

        ---

        ## Race Condition Handling

        **Problem:** Two concurrent requests with same idempotency key.

        **Solution: Redis SET NX (set if not exists)**

        ```python
        def acquire_idempotency_lock(self, idempotency_key: str, merchant_id: str) -> bool:
            """
            Acquire exclusive lock for idempotency key

            Returns:
                True if lock acquired, False if already held
            """
            lock_key = f"idempotency_lock:{merchant_id}:{idempotency_key}"

            # SET NX with 30-second expiry (prevent deadlock)
            acquired = self.redis.set(
                lock_key,
                "locked",
                ex=30,  # Expire after 30 seconds
                nx=True  # Only set if not exists
            )

            return acquired is not None
        ```

        **Usage:**

        ```python
        # First request acquires lock
        if not idempotency_service.acquire_idempotency_lock(idempotency_key, merchant_id):
            # Second concurrent request waits or fails
            raise HTTPException(status_code=409, detail="Request already processing")

        try:
            # Process payment
            payment_intent = await process_payment_intent(payment_data, merchant_id)
        finally:
            # Release lock
            idempotency_service.release_lock(idempotency_key, merchant_id)
        ```

    === "📊 Ledger Design"

        ## The Challenge

        **Problem:** Track money flow accurately, ensure books balance, support audit and reconciliation.

        **Requirements:**

        - **Immutable:** Ledger entries never deleted or modified
        - **Balanced:** Total debits = total credits (always)
        - **Auditable:** Full history for compliance
        - **Real-time:** Up-to-date balances

        ---

        ## Double-Entry Accounting

        **Principle:** Every transaction has two sides (debit and credit). Total debits = total credits.

        **Example: $100 payment with $3 fee**

        | Account | Debit | Credit | Balance After |
        |---------|-------|--------|---------------|
        | Customer Account | $100 | - | $900 |
        | Stripe Holding | - | $100 | $100 |
        | Stripe Holding | $97 | - | $3 |
        | Merchant Account | - | $97 | $97 |
        | Stripe Holding | $3 | - | $0 |
        | Stripe Revenue | - | $3 | $3 |

        **Verification:** $100 + $97 + $3 = $200 debits = $100 + $97 + $3 = $200 credits ✓

        ---

        ## Ledger Service Implementation

        ```python
        from dataclasses import dataclass
        from enum import Enum
        from typing import List
        import psycopg2

        class AccountType(Enum):
            CUSTOMER = "customer"
            MERCHANT = "merchant"
            STRIPE = "stripe"
            STRIPE_REVENUE = "stripe_revenue"
            CARD_NETWORK = "card_network"

        @dataclass
        class LedgerEntry:
            transaction_id: str
            account_id: str
            account_type: AccountType
            debit_amount: int = 0      # Amount in cents
            credit_amount: int = 0     # Amount in cents
            currency: str = "usd"
            description: str = ""

        class LedgerService:
            """Double-entry ledger for payment transactions"""

            def __init__(self, db_connection):
                self.db = db_connection

            def record_payment(
                self,
                transaction_id: str,
                customer_id: str,
                merchant_id: str,
                amount: int,
                fee: int,
                currency: str = "usd"
            ):
                """
                Record payment transaction in double-entry ledger

                Args:
                    transaction_id: Unique transaction identifier
                    customer_id: Customer account ID
                    merchant_id: Merchant account ID
                    amount: Total payment amount in cents
                    fee: Stripe fee in cents
                    currency: Currency code
                """
                merchant_amount = amount - fee

                # Create ledger entries
                entries = [
                    # 1. Customer pays
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id=customer_id,
                        account_type=AccountType.CUSTOMER,
                        debit_amount=amount,
                        description=f"Payment to merchant {merchant_id}"
                    ),
                    # 2. Stripe receives
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_holding",
                        account_type=AccountType.STRIPE,
                        credit_amount=amount,
                        description=f"Payment from customer {customer_id}"
                    ),
                    # 3. Stripe pays merchant
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_holding",
                        account_type=AccountType.STRIPE,
                        debit_amount=merchant_amount,
                        description=f"Payout to merchant {merchant_id}"
                    ),
                    # 4. Merchant receives
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id=merchant_id,
                        account_type=AccountType.MERCHANT,
                        credit_amount=merchant_amount,
                        description=f"Payment from customer {customer_id}"
                    ),
                    # 5. Stripe takes fee
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_holding",
                        account_type=AccountType.STRIPE,
                        debit_amount=fee,
                        description="Processing fee"
                    ),
                    # 6. Stripe revenue
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_revenue",
                        account_type=AccountType.STRIPE_REVENUE,
                        credit_amount=fee,
                        description="Processing fee revenue"
                    )
                ]

                # Insert all entries in single transaction
                self._insert_entries(entries, currency)

                # Verify ledger balance
                self._verify_transaction_balance(transaction_id)

            def record_refund(
                self,
                transaction_id: str,
                original_transaction_id: str,
                customer_id: str,
                merchant_id: str,
                amount: int,
                fee_refund: int,
                currency: str = "usd"
            ):
                """
                Record refund transaction (reverses original payment)

                Args:
                    transaction_id: New refund transaction ID
                    original_transaction_id: Original payment transaction ID
                    customer_id: Customer account ID
                    merchant_id: Merchant account ID
                    amount: Refund amount in cents
                    fee_refund: Fee to refund to merchant (partial)
                    currency: Currency code
                """
                merchant_amount = amount - fee_refund

                entries = [
                    # 1. Customer receives refund
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id=customer_id,
                        account_type=AccountType.CUSTOMER,
                        credit_amount=amount,
                        description=f"Refund from merchant {merchant_id}"
                    ),
                    # 2. Stripe pays customer
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_holding",
                        account_type=AccountType.STRIPE,
                        debit_amount=amount,
                        description=f"Refund to customer {customer_id}"
                    ),
                    # 3. Merchant pays stripe
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id=merchant_id,
                        account_type=AccountType.MERCHANT,
                        debit_amount=merchant_amount,
                        description=f"Refund to customer {customer_id}"
                    ),
                    # 4. Stripe receives from merchant
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_holding",
                        account_type=AccountType.STRIPE,
                        credit_amount=merchant_amount,
                        description=f"Refund from merchant {merchant_id}"
                    ),
                    # 5. Refund fee to merchant
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id="stripe_revenue",
                        account_type=AccountType.STRIPE_REVENUE,
                        debit_amount=fee_refund,
                        description="Fee refund"
                    ),
                    # 6. Merchant receives fee refund
                    LedgerEntry(
                        transaction_id=transaction_id,
                        account_id=merchant_id,
                        account_type=AccountType.MERCHANT,
                        credit_amount=fee_refund,
                        description="Fee refund"
                    )
                ]

                self._insert_entries(entries, currency)
                self._verify_transaction_balance(transaction_id)

            def _insert_entries(self, entries: List[LedgerEntry], currency: str):
                """
                Insert ledger entries atomically with balance updates

                Uses database transaction to ensure atomicity
                """
                cursor = self.db.cursor()

                try:
                    cursor.execute("BEGIN")

                    for entry in entries:
                        # Get current balance
                        cursor.execute(
                            """
                            SELECT balance FROM accounts
                            WHERE id = %s AND currency = %s
                            FOR UPDATE
                            """,
                            (entry.account_id, currency)
                        )
                        result = cursor.fetchone()
                        current_balance = result[0] if result else 0

                        # Calculate new balance
                        if entry.debit_amount > 0:
                            new_balance = current_balance - entry.debit_amount
                        else:
                            new_balance = current_balance + entry.credit_amount

                        # Insert ledger entry
                        cursor.execute(
                            """
                            INSERT INTO ledger_entries (
                                transaction_id, account_id, account_type,
                                debit_amount, credit_amount, currency,
                                balance_after, description, created_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                            )
                            """,
                            (
                                entry.transaction_id,
                                entry.account_id,
                                entry.account_type.value,
                                entry.debit_amount,
                                entry.credit_amount,
                                currency,
                                new_balance,
                                entry.description
                            )
                        )

                        # Update account balance
                        cursor.execute(
                            """
                            INSERT INTO accounts (id, owner_id, account_type, currency, balance)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id, currency)
                            DO UPDATE SET balance = %s, updated_at = NOW()
                            """,
                            (
                                entry.account_id,
                                entry.account_id,  # Owner is same as account for simplicity
                                entry.account_type.value,
                                currency,
                                new_balance,
                                new_balance
                            )
                        )

                    cursor.execute("COMMIT")
                    logger.info(f"Inserted {len(entries)} ledger entries")

                except Exception as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"Ledger insertion failed: {e}")
                    raise

            def _verify_transaction_balance(self, transaction_id: str):
                """
                Verify that debits = credits for this transaction

                Raises exception if imbalanced
                """
                cursor = self.db.cursor()

                cursor.execute(
                    """
                    SELECT
                        SUM(debit_amount) as total_debits,
                        SUM(credit_amount) as total_credits
                    FROM ledger_entries
                    WHERE transaction_id = %s
                    """,
                    (transaction_id,)
                )

                result = cursor.fetchone()
                total_debits = result[0] or 0
                total_credits = result[1] or 0

                if total_debits != total_credits:
                    raise Exception(
                        f"Ledger imbalance for transaction {transaction_id}: "
                        f"debits={total_debits}, credits={total_credits}"
                    )

                logger.info(f"Ledger balanced for transaction {transaction_id}")

            def get_account_balance(self, account_id: str, currency: str = "usd") -> int:
                """
                Get current balance for account

                Returns:
                    Balance in cents
                """
                cursor = self.db.cursor()

                cursor.execute(
                    """
                    SELECT balance FROM accounts
                    WHERE id = %s AND currency = %s
                    """,
                    (account_id, currency)
                )

                result = cursor.fetchone()
                return result[0] if result else 0
        ```

        ---

        ## Ledger Benefits

        | Benefit | How It Helps |
        |---------|--------------|
        | **Accuracy** | Balanced books guarantee correct accounting |
        | **Auditability** | Full transaction history for compliance |
        | **Reconciliation** | Easy to match with bank statements |
        | **Debugging** | Trace money flow for any transaction |
        | **Fraud detection** | Unusual balance changes detected |

    === "🛡️ Fraud Detection"

        ## The Challenge

        **Problem:** Detect fraudulent transactions in real-time (< 50ms) before authorizing payment.

        **Fraud types:**

        - **Stolen cards:** Fraudster uses stolen card details
        - **Account takeover:** Fraudster accesses legitimate account
        - **Card testing:** Testing stolen cards with small transactions
        - **Friendly fraud:** Customer disputes legitimate charge

        ---

        ## Fraud Detection Architecture

        **Two-tier approach:**

        1. **Rule engine:** Fast, deterministic rules (< 10ms)
        2. **ML model:** Probabilistic risk scoring (< 50ms)

        ---

        ## Rule Engine

        ```python
        from dataclasses import dataclass
        from typing import List, Optional
        from enum import Enum

        class RiskLevel(Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            BLOCKED = "blocked"

        @dataclass
        class FraudSignal:
            name: str
            risk_level: RiskLevel
            score: float          # 0.0 to 1.0
            reason: str

        class FraudRuleEngine:
            """Fast rule-based fraud detection"""

            # Thresholds
            MAX_DAILY_TRANSACTIONS = 50
            MAX_DAILY_AMOUNT = 10000_00  # $10,000 in cents
            VELOCITY_WINDOW_SECONDS = 60
            MAX_VELOCITY_TRANSACTIONS = 5

            def __init__(self, redis_client, db_connection):
                self.redis = redis_client
                self.db = db_connection

            def evaluate_payment(
                self,
                payment_data: dict,
                customer_id: str,
                merchant_id: str,
                ip_address: str
            ) -> List[FraudSignal]:
                """
                Evaluate payment for fraud signals

                Args:
                    payment_data: Payment details
                    customer_id: Customer ID
                    merchant_id: Merchant ID
                    ip_address: Client IP address

                Returns:
                    List of fraud signals detected
                """
                signals = []

                # Rule 1: Blocklist check
                if self._is_blocklisted(customer_id, ip_address):
                    signals.append(FraudSignal(
                        name="blocklist",
                        risk_level=RiskLevel.BLOCKED,
                        score=1.0,
                        reason="Customer or IP on blocklist"
                    ))
                    return signals  # Immediate block

                # Rule 2: Velocity check (too many transactions)
                velocity = self._check_velocity(customer_id, merchant_id)
                if velocity > self.MAX_VELOCITY_TRANSACTIONS:
                    signals.append(FraudSignal(
                        name="high_velocity",
                        risk_level=RiskLevel.HIGH,
                        score=0.8,
                        reason=f"{velocity} transactions in {self.VELOCITY_WINDOW_SECONDS}s"
                    ))

                # Rule 3: Daily limits
                daily_stats = self._get_daily_stats(customer_id, merchant_id)
                if daily_stats['count'] > self.MAX_DAILY_TRANSACTIONS:
                    signals.append(FraudSignal(
                        name="daily_transaction_limit",
                        risk_level=RiskLevel.HIGH,
                        score=0.7,
                        reason=f"{daily_stats['count']} transactions today"
                    ))

                if daily_stats['amount'] > self.MAX_DAILY_AMOUNT:
                    signals.append(FraudSignal(
                        name="daily_amount_limit",
                        risk_level=RiskLevel.HIGH,
                        score=0.7,
                        reason=f"${daily_stats['amount']/100} spent today"
                    ))

                # Rule 4: Geographic anomaly
                if self._is_geographic_anomaly(customer_id, ip_address):
                    signals.append(FraudSignal(
                        name="geographic_anomaly",
                        risk_level=RiskLevel.MEDIUM,
                        score=0.5,
                        reason="Transaction from unusual location"
                    ))

                # Rule 5: Card testing pattern
                if self._is_card_testing(merchant_id, ip_address):
                    signals.append(FraudSignal(
                        name="card_testing",
                        risk_level=RiskLevel.HIGH,
                        score=0.8,
                        reason="Multiple small transactions from same IP"
                    ))

                # Rule 6: New customer, high amount
                if self._is_new_customer_high_amount(customer_id, payment_data['amount']):
                    signals.append(FraudSignal(
                        name="new_customer_high_amount",
                        risk_level=RiskLevel.MEDIUM,
                        score=0.6,
                        reason="First transaction over $500"
                    ))

                return signals

            def _is_blocklisted(self, customer_id: str, ip_address: str) -> bool:
                """Check if customer or IP is on blocklist"""
                customer_blocked = self.redis.sismember("blocklist:customers", customer_id)
                ip_blocked = self.redis.sismember("blocklist:ips", ip_address)
                return customer_blocked or ip_blocked

            def _check_velocity(self, customer_id: str, merchant_id: str) -> int:
                """Check transaction velocity (transactions per minute)"""
                key = f"velocity:{customer_id}:{merchant_id}"
                count = self.redis.get(key)
                return int(count) if count else 0

            def _increment_velocity(self, customer_id: str, merchant_id: str):
                """Increment velocity counter"""
                key = f"velocity:{customer_id}:{merchant_id}"
                pipe = self.redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, self.VELOCITY_WINDOW_SECONDS)
                pipe.execute()

            def _get_daily_stats(self, customer_id: str, merchant_id: str) -> dict:
                """Get daily transaction count and amount"""
                today = datetime.utcnow().date()

                cursor = self.db.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(amount), 0)
                    FROM charges
                    WHERE customer_id = %s
                      AND merchant_id = %s
                      AND DATE(created_at) = %s
                      AND status = 'succeeded'
                    """,
                    (customer_id, merchant_id, today)
                )

                result = cursor.fetchone()
                return {'count': result[0], 'amount': result[1]}

            def _is_geographic_anomaly(self, customer_id: str, ip_address: str) -> bool:
                """Detect unusual geographic location"""
                # Get customer's typical country
                typical_country = self._get_typical_country(customer_id)
                if not typical_country:
                    return False  # New customer, no baseline

                # Get current country from IP
                current_country = self._get_country_from_ip(ip_address)

                # Flag if different country
                return typical_country != current_country

            def _is_card_testing(self, merchant_id: str, ip_address: str) -> bool:
                """Detect card testing pattern"""
                # Check for multiple small transactions from same IP
                key = f"card_testing:{merchant_id}:{ip_address}"
                recent_txns = self.redis.get(key)
                return int(recent_txns) > 10 if recent_txns else False

            def _is_new_customer_high_amount(self, customer_id: str, amount: int) -> bool:
                """Flag first transaction over $500"""
                # Check if customer has previous transactions
                cursor = self.db.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM charges
                    WHERE customer_id = %s AND status = 'succeeded'
                    """,
                    (customer_id,)
                )

                count = cursor.fetchone()[0]
                is_new = count == 0
                is_high_amount = amount > 500_00  # $500

                return is_new and is_high_amount
        ```

        ---

        ## Machine Learning Model

        ```python
        import numpy as np
        from typing import Dict

        class FraudMLModel:
            """ML-based fraud risk scoring"""

            def __init__(self, model_path: str):
                # Load pre-trained model (e.g., XGBoost, Random Forest)
                self.model = self._load_model(model_path)

            def predict_fraud_score(self, features: Dict) -> float:
                """
                Predict fraud probability

                Args:
                    features: Feature dictionary

                Returns:
                    Fraud probability (0.0 to 1.0)
                """
                # Extract features
                feature_vector = self._extract_features(features)

                # Predict using ML model
                fraud_probability = self.model.predict_proba(feature_vector)[0][1]

                return fraud_probability

            def _extract_features(self, features: Dict) -> np.ndarray:
                """
                Extract feature vector for ML model

                Features:
                - Transaction amount
                - Customer lifetime value (LTV)
                - Customer age (days since first transaction)
                - Device fingerprint
                - IP address reputation
                - Time of day
                - Day of week
                - Merchant category
                - Card country vs IP country match
                - Velocity features (transactions per hour/day)
                """
                feature_list = [
                    features.get('amount', 0) / 100,  # Normalize to dollars
                    features.get('customer_ltv', 0),
                    features.get('customer_age_days', 0),
                    features.get('device_fingerprint_hash', 0),
                    features.get('ip_reputation_score', 0.5),
                    features.get('hour_of_day', 0),
                    features.get('day_of_week', 0),
                    features.get('merchant_category_code', 0),
                    1 if features.get('card_country') == features.get('ip_country') else 0,
                    features.get('velocity_1h', 0),
                    features.get('velocity_24h', 0),
                ]

                return np.array(feature_list).reshape(1, -1)
        ```

        ---

        ## Combined Fraud Decision

        ```python
        class FraudDetectionService:
            """Orchestrate rule engine and ML model"""

            def __init__(self, rule_engine, ml_model):
                self.rules = rule_engine
                self.ml = ml_model

            def evaluate_payment(
                self,
                payment_data: dict,
                customer_id: str,
                merchant_id: str,
                ip_address: str
            ) -> dict:
                """
                Evaluate payment for fraud

                Returns:
                    {
                        'decision': 'approve' | 'decline' | 'review',
                        'risk_score': 0.0 to 1.0,
                        'signals': [list of fraud signals]
                    }
                """
                # Run rule engine (fast)
                rule_signals = self.rules.evaluate_payment(
                    payment_data, customer_id, merchant_id, ip_address
                )

                # Check for immediate block
                if any(s.risk_level == RiskLevel.BLOCKED for s in rule_signals):
                    return {
                        'decision': 'decline',
                        'risk_score': 1.0,
                        'signals': rule_signals
                    }

                # Extract features for ML
                features = self._extract_ml_features(
                    payment_data, customer_id, merchant_id, ip_address
                )

                # Run ML model
                ml_risk_score = self.ml.predict_fraud_score(features)

                # Combine rule signals and ML score
                rule_risk_score = max([s.score for s in rule_signals], default=0.0)
                combined_risk_score = max(rule_risk_score, ml_risk_score)

                # Make decision
                if combined_risk_score > 0.8:
                    decision = 'decline'
                elif combined_risk_score > 0.5:
                    decision = 'review'  # Manual review
                else:
                    decision = 'approve'

                return {
                    'decision': decision,
                    'risk_score': combined_risk_score,
                    'signals': rule_signals,
                    'ml_score': ml_risk_score
                }
        ```

        ---

        ## Fraud Detection Performance

        | Metric | Target | Actual |
        |--------|--------|--------|
        | **Latency (P95)** | < 50ms | 35ms |
        | **False positive rate** | < 1% | 0.8% |
        | **False negative rate** | < 0.1% | 0.05% |
        | **Fraud caught** | > 95% | 97% |

    === "🔄 Reconciliation"

        ## The Challenge

        **Problem:** Match internal payment records with external card network settlements. Detect discrepancies (missing transactions, amount mismatches).

        **Why needed:**

        - Card networks batch settle transactions (daily)
        - Network may reject transactions after initial approval
        - Chargebacks appear days/weeks later
        - Fees may differ from expected

        ---

        ## Reconciliation Process

        **Daily workflow:**

        1. **Fetch settlement files** from card networks (Visa, Mastercard)
        2. **Match transactions** with internal records
        3. **Identify discrepancies** (missing, duplicate, amount mismatch)
        4. **Generate report** for finance team
        5. **Auto-resolve** simple discrepancies, flag complex ones

        ---

        ## Implementation

        ```python
        from dataclasses import dataclass
        from typing import List, Optional
        from enum import Enum
        import csv

        class DiscrepancyType(Enum):
            MISSING_INTERNAL = "missing_internal"      # In settlement, not in our DB
            MISSING_SETTLEMENT = "missing_settlement"  # In our DB, not in settlement
            AMOUNT_MISMATCH = "amount_mismatch"
            DUPLICATE = "duplicate"
            CHARGEBACK = "chargeback"

        @dataclass
        class SettlementRecord:
            network_txn_id: str
            amount: int
            currency: str
            merchant_id: str
            settlement_date: str
            status: str

        @dataclass
        class Discrepancy:
            type: DiscrepancyType
            network_txn_id: str
            internal_txn_id: Optional[str]
            expected_amount: Optional[int]
            actual_amount: Optional[int]
            details: str

        class ReconciliationService:
            """Daily reconciliation with card networks"""

            def __init__(self, db_connection, s3_client):
                self.db = db_connection
                self.s3 = s3_client

            def reconcile_daily(self, settlement_date: str):
                """
                Reconcile transactions for a given date

                Args:
                    settlement_date: Date to reconcile (YYYY-MM-DD)
                """
                logger.info(f"Starting reconciliation for {settlement_date}")

                # Step 1: Fetch settlement files from card networks
                settlement_records = self._fetch_settlement_files(settlement_date)
                logger.info(f"Fetched {len(settlement_records)} settlement records")

                # Step 2: Fetch internal transactions for same date
                internal_txns = self._fetch_internal_transactions(settlement_date)
                logger.info(f"Fetched {len(internal_txns)} internal transactions")

                # Step 3: Match transactions
                discrepancies = self._match_transactions(
                    settlement_records,
                    internal_txns
                )

                # Step 4: Auto-resolve simple discrepancies
                resolved, unresolved = self._auto_resolve(discrepancies)
                logger.info(
                    f"Resolved {len(resolved)} discrepancies, "
                    f"{len(unresolved)} require manual review"
                )

                # Step 5: Generate report
                self._generate_report(settlement_date, resolved, unresolved)

                # Step 6: Alert if too many discrepancies
                if len(unresolved) > 100:
                    self._alert_finance_team(settlement_date, unresolved)

            def _fetch_settlement_files(self, settlement_date: str) -> List[SettlementRecord]:
                """
                Fetch settlement files from S3 (card networks upload daily)

                Settlement file format (CSV):
                network_txn_id,amount,currency,merchant_id,settlement_date,status
                """
                settlement_records = []

                # Fetch Visa settlement
                visa_file = self.s3.get_object(
                    Bucket='stripe-settlements',
                    Key=f'visa/{settlement_date}.csv'
                )
                settlement_records.extend(self._parse_settlement_file(visa_file['Body']))

                # Fetch Mastercard settlement
                mc_file = self.s3.get_object(
                    Bucket='stripe-settlements',
                    Key=f'mastercard/{settlement_date}.csv'
                )
                settlement_records.extend(self._parse_settlement_file(mc_file['Body']))

                return settlement_records

            def _parse_settlement_file(self, file_content) -> List[SettlementRecord]:
                """Parse CSV settlement file"""
                records = []
                reader = csv.DictReader(file_content.read().decode('utf-8').splitlines())

                for row in reader:
                    records.append(SettlementRecord(
                        network_txn_id=row['network_txn_id'],
                        amount=int(row['amount']),
                        currency=row['currency'],
                        merchant_id=row['merchant_id'],
                        settlement_date=row['settlement_date'],
                        status=row['status']
                    ))

                return records

            def _fetch_internal_transactions(self, settlement_date: str) -> List[dict]:
                """Fetch internal transactions for date"""
                cursor = self.db.cursor()

                cursor.execute(
                    """
                    SELECT
                        c.id,
                        c.network_transaction_id,
                        c.amount,
                        c.currency,
                        pi.merchant_id,
                        c.created_at
                    FROM charges c
                    JOIN payment_intents pi ON c.payment_intent_id = pi.id
                    WHERE DATE(c.created_at) = %s
                      AND c.status = 'succeeded'
                      AND c.captured = TRUE
                    """,
                    (settlement_date,)
                )

                return [
                    {
                        'id': row[0],
                        'network_txn_id': row[1],
                        'amount': row[2],
                        'currency': row[3],
                        'merchant_id': row[4],
                        'created_at': row[5]
                    }
                    for row in cursor.fetchall()
                ]

            def _match_transactions(
                self,
                settlement_records: List[SettlementRecord],
                internal_txns: List[dict]
            ) -> List[Discrepancy]:
                """
                Match settlement records with internal transactions

                Returns:
                    List of discrepancies found
                """
                discrepancies = []

                # Create lookup maps
                settlement_map = {r.network_txn_id: r for r in settlement_records}
                internal_map = {t['network_txn_id']: t for t in internal_txns}

                # Check for missing settlements
                for txn_id, internal_txn in internal_map.items():
                    if txn_id not in settlement_map:
                        discrepancies.append(Discrepancy(
                            type=DiscrepancyType.MISSING_SETTLEMENT,
                            network_txn_id=txn_id,
                            internal_txn_id=internal_txn['id'],
                            expected_amount=internal_txn['amount'],
                            actual_amount=None,
                            details=f"Transaction in DB but not in settlement"
                        ))
                    else:
                        # Check for amount mismatch
                        settlement = settlement_map[txn_id]
                        if settlement.amount != internal_txn['amount']:
                            discrepancies.append(Discrepancy(
                                type=DiscrepancyType.AMOUNT_MISMATCH,
                                network_txn_id=txn_id,
                                internal_txn_id=internal_txn['id'],
                                expected_amount=internal_txn['amount'],
                                actual_amount=settlement.amount,
                                details=f"Amount mismatch: expected ${internal_txn['amount']/100}, got ${settlement.amount/100}"
                            ))

                        # Check for chargebacks
                        if settlement.status == 'chargeback':
                            discrepancies.append(Discrepancy(
                                type=DiscrepancyType.CHARGEBACK,
                                network_txn_id=txn_id,
                                internal_txn_id=internal_txn['id'],
                                expected_amount=internal_txn['amount'],
                                actual_amount=-settlement.amount,
                                details=f"Chargeback received"
                            ))

                # Check for missing internal records
                for txn_id, settlement in settlement_map.items():
                    if txn_id not in internal_map:
                        discrepancies.append(Discrepancy(
                            type=DiscrepancyType.MISSING_INTERNAL,
                            network_txn_id=txn_id,
                            internal_txn_id=None,
                            expected_amount=None,
                            actual_amount=settlement.amount,
                            details=f"Transaction in settlement but not in DB"
                        ))

                return discrepancies

            def _auto_resolve(self, discrepancies: List[Discrepancy]) -> tuple:
                """
                Automatically resolve simple discrepancies

                Returns:
                    (resolved, unresolved) tuple
                """
                resolved = []
                unresolved = []

                for disc in discrepancies:
                    if disc.type == DiscrepancyType.MISSING_SETTLEMENT:
                        # Check if transaction was declined/refunded
                        if self._was_declined_or_refunded(disc.internal_txn_id):
                            resolved.append(disc)
                        else:
                            unresolved.append(disc)

                    elif disc.type == DiscrepancyType.AMOUNT_MISMATCH:
                        # Check if difference is due to currency conversion
                        if abs(disc.expected_amount - disc.actual_amount) < 100:  # $1 tolerance
                            resolved.append(disc)
                        else:
                            unresolved.append(disc)

                    elif disc.type == DiscrepancyType.CHARGEBACK:
                        # Create chargeback record
                        self._create_chargeback(disc)
                        resolved.append(disc)

                    else:
                        unresolved.append(disc)

                return resolved, unresolved

            def _generate_report(
                self,
                settlement_date: str,
                resolved: List[Discrepancy],
                unresolved: List[Discrepancy]
            ):
                """Generate reconciliation report"""
                report = {
                    'date': settlement_date,
                    'summary': {
                        'total_discrepancies': len(resolved) + len(unresolved),
                        'auto_resolved': len(resolved),
                        'requires_review': len(unresolved)
                    },
                    'resolved': [self._discrepancy_to_dict(d) for d in resolved],
                    'unresolved': [self._discrepancy_to_dict(d) for d in unresolved]
                }

                # Save to S3
                self.s3.put_object(
                    Bucket='stripe-reconciliation',
                    Key=f'reports/{settlement_date}.json',
                    Body=json.dumps(report, indent=2)
                )

                logger.info(f"Generated reconciliation report for {settlement_date}")
        ```

        ---

        ## Reconciliation Metrics

        | Metric | Target |
        |--------|--------|
        | **Match rate** | > 99.9% |
        | **Auto-resolution rate** | > 90% |
        | **Reconciliation latency** | < 2 hours after settlement |
        | **False positives** | < 0.1% |

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling payment system from 1M to 100M transactions/day.

    **Scaling challenges at 100M transactions/day:**

    - **Write throughput:** 5,000 write QPS (payments + refunds + ledger)
    - **Read throughput:** 13,600 read QPS (queries + dashboard)
    - **Database transactions:** ACID guarantees at scale
    - **Idempotency:** 100M unique keys per day

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **PostgreSQL writes** | ✅ Yes | Connection pooling (1000 connections), write-ahead log (WAL), read replicas |
    | **Ledger writes** | ✅ Yes | Batch inserts, async ledger updates (eventual consistency acceptable) |
    | **Idempotency cache** | ✅ Yes | Redis cluster (100 nodes), 24-hour TTL, LRU eviction |
    | **Card network latency** | ✅ Yes | Timeout and retry, circuit breaker, fallback to backup processor |
    | **Webhook delivery** | 🟡 Approaching | Kafka partitioning, dedicated webhook worker fleet, exponential backoff |

    ---

    ## Database Scaling

    ### Sharding Strategy

    **Shard by merchant_id:**

    ```
    100M transactions/day
    ├── 10,000 merchants (avg)
    ├── 10,000 transactions/merchant/day
    └── Shard into 100 databases
        ├── Shard 0: merchants 0-99
        ├── Shard 1: merchants 100-199
        └── ...
    ```

    **Benefits:**

    - **Read locality:** Merchant queries hit single shard
    - **Write distribution:** Writes spread across shards
    - **Isolation:** Merchant A's load doesn't affect merchant B

    **Implementation:**

    ```python
    def get_shard_id(merchant_id: str) -> int:
        """Determine database shard for merchant"""
        return hash(merchant_id) % 100  # 100 shards

    def get_db_connection(merchant_id: str):
        """Get database connection for merchant's shard"""
        shard_id = get_shard_id(merchant_id)
        return db_pool[shard_id]
    ```

    ---

    ### Read Replicas

    **Setup:**

    - 1 primary (writes)
    - 3 read replicas (reads)
    - Replication lag: < 100ms

    **Usage:**

    - Dashboard queries: read replicas
    - Transaction queries: read replicas
    - Payment processing: primary (strong consistency)

    ---

    ## Caching Strategy

    | Cache Type | Data | TTL | Size |
    |-----------|------|-----|------|
    | **Idempotency keys** | Request deduplication | 24 hours | 10 GB |
    | **Merchant config** | API keys, settings | 1 hour | 5 GB |
    | **Payment methods** | Tokenized cards | 1 hour | 20 GB |
    | **Fraud features** | Real-time features | 5 minutes | 100 GB |
    | **Rate limiting** | Token buckets | 1 minute | 1 GB |

    **Cache hit rates:**

    - Idempotency: 5% (most requests unique)
    - Merchant config: 95% (rarely changes)
    - Payment methods: 70% (repeated customers)
    - Fraud features: 80% (hot customers)

    ---

    ## Availability & Disaster Recovery

    ### Multi-region Setup

    ```
    Primary Region (US-East)
    ├── API servers: 100 nodes
    ├── PostgreSQL: Primary + 3 replicas
    ├── Redis: 100-node cluster
    └── Kafka: 50 brokers

    Secondary Region (EU-West)
    ├── API servers: 50 nodes
    ├── PostgreSQL: Read replicas (async replication)
    ├── Redis: 100-node cluster
    └── Kafka: 50 brokers

    Disaster Recovery Region (US-West)
    ├── PostgreSQL: Standby (async replication)
    └── Cold backup (activate in 15 minutes)
    ```

    **Failover strategy:**

    - **Database:** Promote read replica to primary (< 1 minute)
    - **API:** Route traffic to secondary region (< 30 seconds)
    - **Redis:** Independent per region (no cross-region dependency)

    ---

    ## Cost Optimization

    **Monthly cost at 100M transactions/day:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $43,200 (150 × m5.2xlarge) |
    | **RDS PostgreSQL** | $86,400 (100 shards × db.r5.2xlarge) |
    | **ElastiCache Redis** | $64,800 (100 nodes) |
    | **MSK Kafka** | $21,600 (50 brokers) |
    | **S3 storage** | $15,000 (7.3 PB) |
    | **Data transfer** | $8,500 (1.2 Gbps egress) |
    | **CloudWatch** | $5,000 (logs + metrics) |
    | **Total** | **$244,500/month** |

    **Revenue (assuming 2.9% + $0.30 per transaction):**

    - 100M transactions/day × $50 avg = $5B/day volume
    - Fee: $5B × 2.9% = $145M/day revenue
    - Monthly: $145M × 30 = $4.35B/month

    **Profit margin:** ~94% (infrastructure is small % of revenue)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Payment Latency (P95)** | < 2s | > 5s |
    | **Payment Success Rate** | > 95% | < 90% |
    | **Database Connection Pool** | < 80% | > 90% |
    | **Idempotency Cache Hit Rate** | > 5% | < 1% |
    | **Webhook Delivery Rate** | > 99% | < 95% |
    | **Fraud False Positive Rate** | < 1% | > 2% |
    | **Ledger Imbalance** | 0 | > 0 |

    **Alerting:**

    - **PagerDuty:** Critical alerts (payment failures, database down)
    - **Slack:** Warning alerts (high latency, cache misses)
    - **Email:** Daily reports (reconciliation, fraud summary)

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Strong consistency:** PostgreSQL with ACID transactions
    2. **Idempotency:** Redis-based deduplication with 24-hour TTL
    3. **Double-entry ledger:** Ensures accurate accounting
    4. **Fraud detection:** Hybrid rule engine + ML model
    5. **Event-driven:** Kafka for webhooks, reconciliation, analytics
    6. **Multi-region:** Active-active with disaster recovery

    ---

    ## Interview Tips

    ✅ **Emphasize consistency** - Money requires strong ACID guarantees

    ✅ **Discuss idempotency** - Critical for payment systems, prevents duplicate charges

    ✅ **Explain ledger design** - Double-entry accounting is industry standard

    ✅ **Cover fraud detection** - Real-time fraud scoring is essential

    ✅ **Mention compliance** - PCI DSS, PSD2, SOC 2 requirements

    ✅ **Plan for failures** - Timeouts, retries, circuit breakers

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent duplicate charges?"** | Idempotency keys with Redis cache, 24-hour TTL, client-generated UUIDs |
    | **"How to ensure ledger accuracy?"** | Double-entry accounting, total debits = total credits, ACID transactions |
    | **"How to detect fraud in real-time?"** | Rule engine (< 10ms) + ML model (< 50ms), combined risk scoring |
    | **"How to handle refunds?"** | Reverse ledger entries, update balances, notify via webhooks |
    | **"How to scale to 1B transactions/day?"** | Shard by merchant, read replicas, cache aggressively, multi-region |
    | **"How to ensure 99.99% availability?"** | Multi-region active-active, database failover, circuit breakers |
    | **"How to reconcile with card networks?"** | Daily batch reconciliation, match on network_transaction_id, auto-resolve discrepancies |

    ---

    ## Security Considerations

    | Concern | Solution |
    |---------|----------|
    | **PCI Compliance** | Tokenize cards, never store raw card numbers, yearly audit |
    | **TLS encryption** | TLS 1.3 for all API traffic, certificate pinning for mobile |
    | **Data encryption** | Encrypt at rest (AES-256), encrypt in transit (TLS) |
    | **API authentication** | Bearer tokens, rate limiting (100 req/sec per merchant) |
    | **Webhook signatures** | HMAC-SHA256 signatures, verify on merchant side |
    | **DDoS protection** | WAF, rate limiting, CloudFlare |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Stripe, PayPal, Square, Adyen, Braintree, Razorpay

---

*Master this problem and you'll be ready for: Financial systems, e-commerce platforms, subscription services, marketplace payments*
