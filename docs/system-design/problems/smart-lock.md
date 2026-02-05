# Design Smart Lock System

A smart lock system that enables keyless entry via Bluetooth/WiFi, virtual key sharing, and remote access control with integration into smart home ecosystems.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐ Medium | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1M locks, 5M users, 10M operations/day (lock/unlock) |
| **Key Challenges** | Bluetooth/WiFi communication, security (encryption), offline mode, battery optimization |
| **Core Concepts** | BLE pairing, virtual keys, access control, end-to-end encryption, offline operation |
| **Companies** | Amazon Key, August Smart Lock, Yale Access, Schlage Encode, Kwikset |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Lock/Unlock** | Control lock via Bluetooth/WiFi/mobile app | P0 (Must have) |
    | **Virtual Keys** | Share temporary/permanent access | P0 (Must have) |
    | **Access Logs** | Audit trail of all lock/unlock events | P0 (Must have) |
    | **Auto-lock/unlock** | Geofencing and proximity-based automation | P1 (Should have) |
    | **Battery Alerts** | Low battery notifications | P0 (Must have) |
    | **Offline Mode** | Bluetooth operation without internet | P0 (Must have) |
    | **Multi-user Access** | Multiple users with different permissions | P0 (Must have) |
    | **Smart Home Integration** | Alexa, Google Home, HomeKit | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Video doorbell integration
    - Package delivery tracking
    - Advanced biometric authentication (fingerprint)
    - Lock hardware manufacturing details
    - Physical security vulnerabilities

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users must access homes reliably |
    | **Latency (Lock/Unlock)** | < 2s response time | Fast operation critical for user experience |
    | **Latency (Bluetooth)** | < 500ms via BLE | Local operation must be instant |
    | **Security** | End-to-end encryption | Prevent unauthorized access |
    | **Battery Life** | 6-12 months | Minimize battery replacements |
    | **Offline Operation** | 100% Bluetooth functionality | Must work without internet |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total locks: 1M smart locks deployed
    Total users: 5M registered users (5 users per lock average)
    Daily operations: 10M lock/unlock operations/day

    Lock/unlock frequency:
    - Average per lock: 10 operations/day
    - Peak hours: 7-9am, 5-7pm (3x multiplier)
    - Operations per second (average): 10M / 86,400 = 115 ops/sec
    - Peak: 345 ops/sec

    Virtual key operations:
    - Key shares: 500K/day (1 per 2 locks)
    - Key revocations: 200K/day
    - Key checks: 5M/day (sync operations)

    Access log queries:
    - Log writes: 10M/day (every operation logged)
    - Log reads: 2M queries/day
    ```

    ### Storage Estimates

    ```
    Lock data:
    - Per lock: 2 KB (lock_id, owner_id, settings, firmware_version)
    - Total: 1M × 2 KB = 2 GB

    User data:
    - Per user: 5 KB (user_id, name, email, phone, preferences)
    - Total: 5M × 5 KB = 25 GB

    Virtual keys:
    - Per key: 1 KB (key_id, lock_id, user_id, permissions, expiry)
    - Keys per lock: 5 average
    - Total: 1M × 5 × 1 KB = 5 GB

    Access logs:
    - Per log: 500 bytes (lock_id, user_id, action, timestamp, method)
    - Daily: 10M × 500 bytes = 5 GB/day
    - Retention (2 years): 5 GB × 730 = 3.65 TB

    Encryption keys:
    - Per lock: 256 bytes (symmetric key, public key)
    - Total: 1M × 256 bytes = 256 MB

    Total: 2 GB (locks) + 25 GB (users) + 5 GB (keys) + 3.65 TB (logs) ≈ 3.7 TB
    ```

    ### Memory Estimates (Caching)

    ```
    Active sessions:
    - Concurrent users: 50K (1% of users active)
    - Session data: 10 KB per user
    - Total: 50K × 10 KB = 500 MB

    Lock state cache:
    - Active locks: 100K (10% active in peak hours)
    - State data: 5 KB per lock
    - Total: 100K × 5 KB = 500 MB

    Virtual key cache (hot keys):
    - Frequently accessed: 500K keys
    - Per key: 1 KB
    - Total: 500 MB

    Total cache: 1.5 GB
    ```

    ---

    ## Key Assumptions

    1. Average lock/unlock: 10 operations/day per lock
    2. Battery life: 6-12 months (depends on operation frequency)
    3. Bluetooth range: 10-15 meters
    4. WiFi connectivity: 80% of locks have WiFi bridge
    5. Virtual key usage: 5 keys per lock average
    6. Access log retention: 2 years
    7. Firmware updates: Monthly

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Security first** - End-to-end encryption for all communications
    2. **Offline-first** - Bluetooth operation without cloud dependency
    3. **Battery optimization** - Minimize radio usage and processing
    4. **Multi-protocol** - BLE for local, WiFi for remote, cloud for sync
    5. **Access control** - Granular permissions and time-based restrictions

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App<br/>iOS/Android]
            Voice[Voice Assistant<br/>Alexa/Google]
            Watch[Smart Watch]
        end

        subgraph "Communication Protocols"
            BLE[Bluetooth LE<br/>Direct connection]
            WiFi_Bridge[WiFi Bridge<br/>August Connect]
        end

        subgraph "API Gateway"
            LB[Load Balancer]
            API_GW[API Gateway<br/>Auth, Rate limiting]
        end

        subgraph "Core Services"
            Lock_Service[Lock Service<br/>Lock operations]
            Key_Service[Key Management<br/>Virtual keys]
            Access_Service[Access Control<br/>Permissions]
            Auth_Service[Auth Service<br/>User authentication]
            Notification[Notification Service<br/>Push/Email/SMS]
            Firmware_Service[Firmware Service<br/>OTA updates]
        end

        subgraph "Lock Device"
            Lock_Firmware[Lock Firmware<br/>BLE stack]
            Motor[Motor Controller<br/>Lock/unlock]
            Battery[Battery Monitor]
            Sensors[Door sensor<br/>Position detection]
        end

        subgraph "Caching"
            Redis_Session[Redis<br/>User sessions]
            Redis_Lock[Redis<br/>Lock state]
            Redis_Key[Redis<br/>Virtual keys hot cache]
        end

        subgraph "Storage"
            Lock_DB[(Lock DB<br/>PostgreSQL<br/>Locks, settings)]
            User_DB[(User DB<br/>PostgreSQL<br/>Users, keys)]
            Log_DB[(Access Logs<br/>Cassandra<br/>Time-series logs)]
            KMS[(Key Management<br/>AWS KMS<br/>Encryption keys)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Events, logs]
        end

        subgraph "External Services"
            Push_Service[FCM/APNs<br/>Push notifications]
            Email_Service[SendGrid<br/>Email notifications]
            Smart_Home[Smart Home APIs<br/>Alexa, Google]
        end

        Mobile --> BLE
        Mobile --> LB
        Watch --> BLE
        Voice --> Smart_Home
        Smart_Home --> LB

        BLE <--> Lock_Firmware
        WiFi_Bridge --> LB
        WiFi_Bridge <--> Lock_Firmware

        LB --> API_GW
        API_GW --> Lock_Service
        API_GW --> Key_Service
        API_GW --> Access_Service
        API_GW --> Auth_Service

        Lock_Service --> Redis_Lock
        Lock_Service --> Lock_DB
        Lock_Service --> Kafka

        Key_Service --> Redis_Key
        Key_Service --> User_DB
        Key_Service --> KMS

        Access_Service --> Log_DB
        Access_Service --> Kafka

        Auth_Service --> User_DB
        Auth_Service --> Redis_Session

        Kafka --> Notification
        Notification --> Push_Service
        Notification --> Email_Service

        Lock_Firmware --> Motor
        Lock_Firmware --> Battery
        Lock_Firmware --> Sensors

        Firmware_Service --> Lock_Firmware

        style BLE fill:#e8f5e9
        style WiFi_Bridge fill:#e8f5e9
        style Redis_Session fill:#fff4e1
        style Redis_Lock fill:#fff4e1
        style Redis_Key fill:#fff4e1
        style Lock_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Log_DB fill:#ffe1e1
        style KMS fill:#ffebee
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Bluetooth LE** | Low power, offline operation, direct device connection | WiFi only (higher power, requires internet), NFC (very short range) |
    | **WiFi Bridge** | Remote access when user away from home | Cellular (expensive), LoRaWAN (limited availability) |
    | **AWS KMS** | Hardware-backed encryption key storage | Self-managed keys (less secure), application-level only (vulnerable) |
    | **Cassandra (Logs)** | High write throughput for access logs (115 ops/sec) | PostgreSQL (adequate but less scalable), DynamoDB (vendor lock-in) |
    | **Redis (Lock State)** | Fast lock state lookup (< 10ms) | Database only (slower, higher load) |
    | **Kafka** | Reliable event streaming for audit trail | Direct DB writes (no replay capability), SQS (limited ordering) |

    **Key Trade-off:** We chose **Bluetooth-first architecture** for offline reliability but **WiFi bridge for convenience**. This dual approach balances security (local control) with usability (remote access).

    ---

    ## API Design

    ### 1. Lock/Unlock Operation

    **Request:**
    ```http
    POST /api/v1/locks/{lock_id}/operate
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "action": "unlock",  // lock, unlock
      "method": "mobile_app",  // mobile_app, auto_unlock, voice, remote
      "location": {
        "lat": 37.7749,
        "lng": -122.4194
      },
      "bluetooth_proximity": true,
      "challenge_response": "encrypted_response_token"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "operation_id": "op_abc123",
      "lock_id": "lock_xyz789",
      "action": "unlock",
      "status": "success",  // success, failed, timeout
      "timestamp": "2026-02-05T10:30:00Z",
      "battery_level": 75,
      "door_state": "closed"  // open, closed, unknown
    }
    ```

    **Design Notes:**

    - Include challenge-response for security (prevent replay attacks)
    - Return battery level with every operation
    - Log all operations for audit trail
    - Support both synchronous (via WiFi) and asynchronous (via Bluetooth polling)

    ---

    ### 2. Create Virtual Key

    **Request:**
    ```http
    POST /api/v1/locks/{lock_id}/keys
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "recipient_email": "guest@example.com",
      "key_name": "Guest Key - John",
      "key_type": "temporary",  // permanent, temporary, recurring
      "permissions": {
        "lock": true,
        "unlock": true,
        "view_logs": false,
        "manage_keys": false
      },
      "schedule": {
        "start_time": "2026-02-05T00:00:00Z",
        "end_time": "2026-02-07T23:59:59Z",
        "days_of_week": ["monday", "tuesday", "wednesday"],
        "time_windows": [
          {"start": "09:00", "end": "17:00"}
        ]
      },
      "access_limit": 5  // Max uses (optional)
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "key_id": "key_abc123",
      "lock_id": "lock_xyz789",
      "recipient_email": "guest@example.com",
      "key_type": "temporary",
      "status": "active",  // pending, active, expired, revoked
      "created_at": "2026-02-05T10:30:00Z",
      "expires_at": "2026-02-07T23:59:59Z",
      "invitation_link": "https://app.smartlock.com/keys/accept/abc123",
      "access_code": "1234-5678"  // Optional PIN code
    }
    ```

    **Design Notes:**

    - Generate cryptographic key pair for virtual key
    - Send invitation via email/SMS
    - Support recurring schedules (e.g., cleaning service every Monday)
    - Access limits for one-time deliveries

    ---

    ### 3. Get Access Logs

    **Request:**
    ```http
    GET /api/v1/locks/{lock_id}/logs?start_time=2026-02-01T00:00:00Z&end_time=2026-02-05T23:59:59Z&limit=50&offset=0
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "logs": [
        {
          "log_id": "log_abc123",
          "lock_id": "lock_xyz789",
          "user_id": "user_123",
          "user_name": "John Doe",
          "action": "unlock",
          "method": "mobile_app",  // mobile_app, auto_unlock, voice, keypad, virtual_key
          "status": "success",
          "timestamp": "2026-02-05T10:30:00Z",
          "location": {
            "lat": 37.7749,
            "lng": -122.4194
          },
          "battery_level": 75
        },
        // ... more logs
      ],
      "pagination": {
        "total": 1250,
        "limit": 50,
        "offset": 0,
        "has_more": true
      }
    }
    ```

    ---

    ### 4. Bluetooth Direct Operation (Local Protocol)

    **Lock Command (App → Lock via BLE):**
    ```
    Protocol: Custom binary protocol over BLE GATT

    Frame structure:
    [HEADER][COMMAND][PAYLOAD][SIGNATURE]

    HEADER (4 bytes):
    - Protocol version: 1 byte
    - Message type: 1 byte (0x01=command, 0x02=response)
    - Payload length: 2 bytes

    COMMAND (1 byte):
    - 0x10: Unlock
    - 0x11: Lock
    - 0x20: Get status
    - 0x30: Sync virtual keys

    PAYLOAD (variable):
    - User ID: 16 bytes (UUID)
    - Timestamp: 8 bytes (Unix ms)
    - Nonce: 16 bytes (prevent replay)
    - Key ID: 16 bytes (virtual key)

    SIGNATURE (32 bytes):
    - HMAC-SHA256 of entire message
    - Key: Shared secret from pairing
    ```

    **Example Unlock Command:**
    ```
    Header:  [01][01][00 38][...]
    Command: [10]
    Payload: [user_id (16)] [timestamp (8)] [nonce (16)] [key_id (16)]
    Signature: [hmac_sha256 (32)]
    ```

    **Lock Response:**
    ```
    Header:  [01][02][00 10][...]
    Status:  [00] (success)
    Battery: [4B] (75%)
    Door:    [01] (closed)
    Timestamp: [timestamp (8)]
    Signature: [hmac_sha256 (32)]
    ```

    ---

    ## Database Schema

    ### Locks (PostgreSQL)

    ```sql
    -- Locks table
    CREATE TABLE locks (
        lock_id UUID PRIMARY KEY,
        serial_number VARCHAR(50) UNIQUE NOT NULL,
        owner_id UUID NOT NULL REFERENCES users(user_id),
        lock_name VARCHAR(100),
        model VARCHAR(50),
        firmware_version VARCHAR(20),
        hardware_version VARCHAR(20),
        installation_date DATE,

        -- Settings
        auto_lock_enabled BOOLEAN DEFAULT true,
        auto_lock_delay_seconds INT DEFAULT 30,
        auto_unlock_enabled BOOLEAN DEFAULT false,
        geofence_radius_meters INT DEFAULT 100,

        -- State (cache from device)
        last_seen_at TIMESTAMP,
        battery_level INT,  -- 0-100
        door_state VARCHAR(20),  -- open, closed, jammed, unknown
        lock_state VARCHAR(20),  -- locked, unlocked, unknown

        -- Security
        encryption_key_id VARCHAR(100),  -- Reference to KMS
        pairing_secret_hash BYTEA,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_owner (owner_id),
        INDEX idx_serial (serial_number)
    );

    -- WiFi bridge/gateway
    CREATE TABLE lock_bridges (
        bridge_id UUID PRIMARY KEY,
        lock_id UUID REFERENCES locks(lock_id),
        mac_address VARCHAR(17) UNIQUE,
        ip_address INET,
        connection_status VARCHAR(20),  -- online, offline
        last_heartbeat TIMESTAMP,

        INDEX idx_lock (lock_id)
    );
    ```

    ---

    ### Virtual Keys (PostgreSQL)

    ```sql
    -- Virtual keys table
    CREATE TABLE virtual_keys (
        key_id UUID PRIMARY KEY,
        lock_id UUID NOT NULL REFERENCES locks(lock_id),
        issued_by_user_id UUID NOT NULL REFERENCES users(user_id),
        recipient_user_id UUID REFERENCES users(user_id),
        recipient_email VARCHAR(255),

        key_name VARCHAR(100),
        key_type VARCHAR(20) NOT NULL,  -- permanent, temporary, recurring
        status VARCHAR(20) NOT NULL,  -- pending, active, expired, revoked

        -- Permissions
        can_lock BOOLEAN DEFAULT true,
        can_unlock BOOLEAN DEFAULT true,
        can_view_logs BOOLEAN DEFAULT false,
        can_manage_keys BOOLEAN DEFAULT false,

        -- Schedule
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        days_of_week INT[],  -- 0=Sunday, 6=Saturday
        time_windows JSONB,  -- [{"start": "09:00", "end": "17:00"}]

        -- Usage tracking
        access_limit INT,  -- NULL = unlimited
        access_count INT DEFAULT 0,
        last_used_at TIMESTAMP,

        -- Security
        encryption_key_id VARCHAR(100),
        invitation_token VARCHAR(100) UNIQUE,
        access_code VARCHAR(20),  -- Optional PIN

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        revoked_at TIMESTAMP,

        INDEX idx_lock (lock_id, status),
        INDEX idx_recipient (recipient_user_id),
        INDEX idx_invitation (invitation_token)
    );

    -- Ensure key validity
    CREATE INDEX idx_active_keys ON virtual_keys (lock_id, status)
    WHERE status = 'active';
    ```

    ---

    ### Access Logs (Cassandra)

    ```sql
    -- Time-series access logs
    CREATE TABLE access_logs (
        lock_id UUID,
        log_time TIMESTAMP,
        log_id UUID,
        user_id UUID,
        key_id UUID,

        action VARCHAR(20),  -- lock, unlock, auto_lock, auto_unlock
        method VARCHAR(20),  -- mobile_app, virtual_key, voice, auto, keypad
        status VARCHAR(20),  -- success, failed, denied, timeout

        -- Context
        battery_level INT,
        door_state VARCHAR(20),
        failure_reason TEXT,

        -- Security audit
        ip_address TEXT,
        device_id TEXT,
        location_lat DOUBLE,
        location_lng DOUBLE,

        PRIMARY KEY (lock_id, log_time, log_id)
    ) WITH CLUSTERING ORDER BY (log_time DESC);

    -- Query pattern: Get recent logs for specific lock
    -- SELECT * FROM access_logs WHERE lock_id = ? AND log_time > ? LIMIT 100;

    -- User activity logs (separate table for user-centric queries)
    CREATE TABLE user_access_logs (
        user_id UUID,
        log_time TIMESTAMP,
        log_id UUID,
        lock_id UUID,
        action VARCHAR(20),
        status VARCHAR(20),

        PRIMARY KEY (user_id, log_time, log_id)
    ) WITH CLUSTERING ORDER BY (log_time DESC);
    ```

    ---

    ## Data Flow Diagrams

    ### Lock/Unlock Flow (Bluetooth)

    ```mermaid
    sequenceDiagram
        participant User
        participant Mobile_App
        participant Lock_BLE
        participant Cloud
        participant Notification

        User->>Mobile_App: Press unlock button
        Mobile_App->>Mobile_App: Check cached virtual key

        alt Bluetooth in range
            Mobile_App->>Lock_BLE: BLE: Unlock command + signature
            Lock_BLE->>Lock_BLE: Verify signature & permissions
            Lock_BLE->>Lock_BLE: Check schedule & access limit

            alt Authorized
                Lock_BLE->>Lock_BLE: Unlock motor
                Lock_BLE-->>Mobile_App: Success + battery level
                Mobile_App-->>User: Unlocked!

                Mobile_App->>Cloud: Log operation (async)
                Cloud->>Notification: Notify other users
            else Denied
                Lock_BLE-->>Mobile_App: Access denied
                Mobile_App-->>User: Show error
            end
        else No Bluetooth (Remote unlock)
            Mobile_App->>Cloud: POST /locks/{id}/operate
            Cloud->>Cloud: Verify permissions
            Cloud->>Lock_BLE: WiFi bridge: Unlock command
            Lock_BLE->>Lock_BLE: Unlock motor
            Lock_BLE-->>Cloud: Success
            Cloud-->>Mobile_App: Operation confirmed
            Mobile_App-->>User: Unlocked!
        end
    ```

    **Flow Explanation:**

    1. **Local-first** - Try Bluetooth direct connection first (< 500ms)
    2. **Signature verification** - Lock verifies HMAC to prevent replay attacks
    3. **Permission check** - Validate virtual key schedule and limits
    4. **Motor control** - Physical unlock mechanism
    5. **Async logging** - Log to cloud when connectivity available
    6. **Fallback to remote** - Use WiFi bridge if Bluetooth unavailable

    ---

    ### Virtual Key Sharing Flow

    ```mermaid
    sequenceDiagram
        participant Owner
        participant API
        participant Key_Service
        participant KMS
        participant Email_Service
        participant Guest
        participant Lock

        Owner->>API: POST /locks/{id}/keys (guest details)
        API->>Key_Service: Create virtual key

        Key_Service->>KMS: Generate encryption key pair
        KMS-->>Key_Service: Public/private keys

        Key_Service->>DB: INSERT virtual_key (encrypted)
        Key_Service->>Email_Service: Send invitation email
        Email_Service->>Guest: Email with invitation link

        Guest->>API: Accept invitation (GET /keys/accept/{token})
        API->>Key_Service: Activate key
        Key_Service->>DB: UPDATE status = active

        Key_Service->>Guest: Return key credentials
        Guest->>Lock: Sync keys via Bluetooth
        Lock->>Lock: Store encrypted key in flash memory
        Lock-->>Guest: Key synced, ready to use
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Smart Lock subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Bluetooth Communication** | How to secure BLE connection? | Encrypted pairing + challenge-response |
    | **Offline Operation** | How to work without internet? | Local key storage + cryptographic verification |
    | **Battery Optimization** | How to last 6-12 months? | Low-power BLE, sleep modes, efficient crypto |
    | **Security & Encryption** | How to prevent unauthorized access? | E2E encryption, key rotation, audit logging |

    ---

    === "📡 Bluetooth Communication"

        ## The Challenge

        **Problem:** Secure, low-latency communication between mobile app and lock over Bluetooth LE.

        **Requirements:**
        - Low latency: < 500ms for lock/unlock
        - Low power: Minimize battery drain
        - Security: Prevent eavesdropping and replay attacks
        - Range: 10-15 meters typical

        ---

        ## BLE Architecture

        **Bluetooth LE GATT Profile:**

        ```
        Smart Lock Service (UUID: custom)
        ├── Lock Control Characteristic (Write)
        │   └── Commands: lock, unlock, get_status
        ├── Lock Status Characteristic (Read/Notify)
        │   └── Battery, door state, lock state
        ├── Key Management Characteristic (Write)
        │   └── Sync virtual keys
        └── Event Log Characteristic (Read)
            └── Recent operations (cached)
        ```

        **Implementation:**

        ```python
        import asyncio
        from bleak import BleakClient
        import hmac
        import hashlib
        import time
        import struct

        class SmartLockBLEClient:
            """Bluetooth LE client for smart lock communication"""

            # GATT Service and Characteristic UUIDs
            LOCK_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
            LOCK_CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef1"
            LOCK_STATUS_UUID = "12345678-1234-5678-1234-56789abcdef2"

            COMMAND_UNLOCK = 0x10
            COMMAND_LOCK = 0x11
            COMMAND_GET_STATUS = 0x20

            def __init__(self, lock_mac_address: str, shared_secret: bytes):
                self.mac_address = lock_mac_address
                self.shared_secret = shared_secret  # From pairing process
                self.client = None

            async def connect(self):
                """Establish BLE connection to lock"""
                self.client = BleakClient(self.mac_address)
                await self.client.connect()
                print(f"Connected to lock: {self.mac_address}")

            async def unlock(self, user_id: str, key_id: str) -> dict:
                """
                Send unlock command to lock

                Args:
                    user_id: User identifier
                    key_id: Virtual key identifier

                Returns:
                    {status, battery_level, door_state, timestamp}
                """
                # Build command payload
                command = self._build_command(
                    self.COMMAND_UNLOCK,
                    user_id,
                    key_id
                )

                # Send command via BLE
                await self.client.write_gatt_char(self.LOCK_CONTROL_UUID, command)

                # Wait for response (notify)
                response = await self._wait_for_response(timeout=5.0)

                return self._parse_response(response)

            def _build_command(self, command_type: int, user_id: str, key_id: str) -> bytes:
                """
                Build encrypted command packet

                Frame: [HEADER][COMMAND][PAYLOAD][SIGNATURE]
                """
                # Header (4 bytes)
                protocol_version = 0x01
                message_type = 0x01  # Command
                payload_length = 56  # user_id(16) + timestamp(8) + nonce(16) + key_id(16)
                header = struct.pack('!BBH', protocol_version, message_type, payload_length)

                # Command (1 byte)
                command = struct.pack('!B', command_type)

                # Payload
                user_id_bytes = bytes.fromhex(user_id.replace('-', ''))[:16]
                timestamp = int(time.time() * 1000)
                timestamp_bytes = struct.pack('!Q', timestamp)
                nonce = os.urandom(16)  # Random nonce to prevent replay
                key_id_bytes = bytes.fromhex(key_id.replace('-', ''))[:16]

                payload = user_id_bytes + timestamp_bytes + nonce + key_id_bytes

                # Signature (HMAC-SHA256)
                message = header + command + payload
                signature = hmac.new(self.shared_secret, message, hashlib.sha256).digest()

                return message + signature

            def _verify_response_signature(self, response: bytes) -> bool:
                """Verify response signature from lock"""
                if len(response) < 32:
                    return False

                message = response[:-32]
                signature = response[-32:]

                expected_signature = hmac.new(
                    self.shared_secret,
                    message,
                    hashlib.sha256
                ).digest()

                return hmac.compare_digest(signature, expected_signature)

            async def _wait_for_response(self, timeout: float) -> bytes:
                """Wait for lock response via notification"""
                response = None

                def notification_handler(sender, data):
                    nonlocal response
                    response = data

                # Subscribe to status characteristic
                await self.client.start_notify(self.LOCK_STATUS_UUID, notification_handler)

                # Wait for response
                start_time = time.time()
                while response is None and (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)

                await self.client.stop_notify(self.LOCK_STATUS_UUID)

                if response is None:
                    raise TimeoutError("No response from lock")

                # Verify signature
                if not self._verify_response_signature(response):
                    raise SecurityError("Invalid response signature")

                return response

            def _parse_response(self, response: bytes) -> dict:
                """Parse lock response"""
                # Response format: [HEADER][STATUS][BATTERY][DOOR_STATE][TIMESTAMP][SIGNATURE]

                # Skip header (4 bytes)
                offset = 4

                # Status (1 byte): 0x00=success, 0x01=denied, 0x02=error
                status_code = response[offset]
                offset += 1

                # Battery level (1 byte): 0-100%
                battery_level = response[offset]
                offset += 1

                # Door state (1 byte): 0x00=closed, 0x01=open, 0x02=unknown
                door_state_code = response[offset]
                offset += 1

                # Timestamp (8 bytes)
                timestamp = struct.unpack('!Q', response[offset:offset+8])[0]

                return {
                    'status': 'success' if status_code == 0x00 else 'denied',
                    'battery_level': battery_level,
                    'door_state': ['closed', 'open', 'unknown'][door_state_code],
                    'timestamp': timestamp
                }

            async def disconnect(self):
                """Close BLE connection"""
                if self.client:
                    await self.client.disconnect()
        ```

        ---

        ## Pairing Process

        **Initial pairing (one-time):**

        1. **User initiates pairing** - Press button on lock
        2. **Lock enters pairing mode** - BLE advertising with pairing flag
        3. **App discovers lock** - Scan for BLE devices
        4. **Secure pairing** - BLE Secure Connections (ECDH key exchange)
        5. **Exchange credentials** - App sends user ID, lock generates shared secret
        6. **Store secret** - Both sides store shared secret for future authentication
        7. **Register with cloud** - App registers lock with cloud backend

        **Security:**
        - Use BLE Secure Connections (Elliptic Curve Diffie-Hellman)
        - Shared secret never transmitted, derived from ECDH
        - Lock only accepts pairing when button pressed (prevents remote attacks)

        ---

        ## Power Optimization

        **BLE power modes:**

        ```c
        // Lock firmware (C/embedded)

        // BLE advertising parameters
        #define ADV_INTERVAL_MS 1000  // Advertise every 1 second
        #define CONNECTION_INTERVAL_MS 100  // 100ms when connected
        #define SLAVE_LATENCY 4  // Skip 4 intervals to save power

        void ble_init() {
            // Configure low-power BLE parameters
            ble_gap_adv_params_t adv_params = {
                .interval = ADV_INTERVAL_MS,
                .timeout = 0,  // No timeout
                .tx_power = -8  // Reduced TX power for shorter range
            };

            // Connection parameters
            ble_gap_conn_params_t conn_params = {
                .min_conn_interval = CONNECTION_INTERVAL_MS,
                .max_conn_interval = CONNECTION_INTERVAL_MS,
                .slave_latency = SLAVE_LATENCY,  // Wake every 5th interval
                .conn_sup_timeout = 4000  // 4 second supervision timeout
            };
        }

        // Sleep between operations
        void lock_sleep() {
            // Power down unused peripherals
            disable_uart();
            disable_spi();

            // Enter low-power mode
            nrf_pwr_mgmt_run();  // Nordic nRF52 power management
        }
        ```

        **Battery life calculation:**

        ```
        Battery capacity: 1000 mAh (4x AA batteries)

        Power consumption:
        - Sleep mode: 5 µA
        - BLE advertising: 15 mA (1% duty cycle) = 150 µA average
        - BLE connected: 10 mA (10% duty cycle) = 1 mA average
        - Motor operation: 500 mA (1 second) = 14 µA average (10 ops/day)
        - Door sensor: 10 µA

        Total average: 5 + 150 + 1 + 14 + 10 = 180 µA

        Battery life: 1000 mAh / 0.18 mA = 5555 hours ≈ 231 days (7.7 months)

        With optimizations (reduce advertising, sleep mode):
        - Reduced to 100 µA average
        - Battery life: 10,000 hours ≈ 416 days (13.8 months)
        ```

    === "🔒 Security & Encryption"

        ## The Challenge

        **Problem:** Prevent unauthorized access while maintaining usability.

        **Threat model:**
        - Eavesdropping on BLE communication
        - Replay attacks (captured unlock command)
        - Stolen/lost phone (access to virtual keys)
        - Compromised cloud backend
        - Physical attacks on lock device

        ---

        ## End-to-End Encryption

        **Key hierarchy:**

        ```
        Master Key (AWS KMS)
        └── Lock Master Key (per lock, 256-bit AES)
            ├── Pairing Secret (per device, 256-bit)
            │   └── Session Keys (per connection, ephemeral)
            └── Virtual Key Secrets (per key, 256-bit)
        ```

        **Implementation:**

        ```python
        import boto3
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        import os

        class LockSecurityManager:
            """Manage encryption keys and security operations"""

            def __init__(self):
                self.kms_client = boto3.client('kms')
                self.kms_key_id = 'arn:aws:kms:us-east-1:123456789:key/...'

            def create_lock_master_key(self, lock_id: str) -> str:
                """
                Generate master key for lock (stored in KMS)

                Returns:
                    KMS key ID reference
                """
                # Generate data key from KMS
                response = self.kms_client.generate_data_key(
                    KeyId=self.kms_key_id,
                    KeySpec='AES_256'
                )

                # Store encrypted key in database
                encrypted_key = response['CiphertextBlob']
                plaintext_key = response['Plaintext']

                # Store reference
                key_id = self._store_encrypted_key(lock_id, encrypted_key)

                return key_id

            def create_virtual_key(
                self,
                lock_id: str,
                key_id: str,
                permissions: dict
            ) -> bytes:
                """
                Create virtual key credentials (encrypted)

                Returns:
                    Encrypted key data for mobile app
                """
                # Get lock master key
                lock_master_key = self._get_lock_master_key(lock_id)

                # Generate virtual key secret
                virtual_key_secret = os.urandom(32)  # 256-bit

                # Create key payload
                key_data = {
                    'key_id': key_id,
                    'lock_id': lock_id,
                    'permissions': permissions,
                    'secret': virtual_key_secret.hex(),
                    'version': 1
                }

                # Encrypt with lock master key
                encrypted_key = self._encrypt_aes_gcm(
                    json.dumps(key_data).encode(),
                    lock_master_key
                )

                return encrypted_key

            def _encrypt_aes_gcm(self, plaintext: bytes, key: bytes) -> bytes:
                """
                Encrypt data with AES-256-GCM (authenticated encryption)

                Returns:
                    iv (12 bytes) + ciphertext + tag (16 bytes)
                """
                # Generate random IV
                iv = os.urandom(12)

                # Encrypt
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv)
                )
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(plaintext) + encryptor.finalize()

                # Return IV + ciphertext + tag
                return iv + ciphertext + encryptor.tag

            def _decrypt_aes_gcm(self, encrypted: bytes, key: bytes) -> bytes:
                """Decrypt AES-256-GCM"""
                # Extract components
                iv = encrypted[:12]
                tag = encrypted[-16:]
                ciphertext = encrypted[12:-16]

                # Decrypt
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv, tag)
                )
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(ciphertext) + decryptor.finalize()

                return plaintext

            def verify_lock_operation(
                self,
                lock_id: str,
                key_id: str,
                signature: bytes,
                message: bytes
            ) -> bool:
                """
                Verify operation signature

                Args:
                    lock_id: Lock identifier
                    key_id: Virtual key identifier
                    signature: HMAC signature
                    message: Command message

                Returns:
                    True if valid signature
                """
                # Get virtual key secret
                virtual_key = self._get_virtual_key(key_id)
                if not virtual_key or virtual_key['lock_id'] != lock_id:
                    return False

                # Check key status and schedule
                if not self._is_key_valid(virtual_key):
                    return False

                # Verify HMAC signature
                secret = bytes.fromhex(virtual_key['secret'])
                expected_signature = hmac.new(
                    secret,
                    message,
                    hashlib.sha256
                ).digest()

                return hmac.compare_digest(signature, expected_signature)

            def _is_key_valid(self, virtual_key: dict) -> bool:
                """Check if virtual key is currently valid"""
                now = datetime.utcnow()

                # Check status
                if virtual_key['status'] != 'active':
                    return False

                # Check expiry
                if virtual_key.get('end_time'):
                    if now > virtual_key['end_time']:
                        return False

                # Check schedule (day of week, time windows)
                if virtual_key.get('days_of_week'):
                    if now.weekday() not in virtual_key['days_of_week']:
                        return False

                if virtual_key.get('time_windows'):
                    current_time = now.time()
                    in_window = False
                    for window in virtual_key['time_windows']:
                        start = datetime.strptime(window['start'], '%H:%M').time()
                        end = datetime.strptime(window['end'], '%H:%M').time()
                        if start <= current_time <= end:
                            in_window = True
                            break
                    if not in_window:
                        return False

                # Check access limit
                if virtual_key.get('access_limit'):
                    if virtual_key['access_count'] >= virtual_key['access_limit']:
                        return False

                return True
        ```

        ---

        ## Replay Attack Prevention

        **Problem:** Attacker captures unlock command and replays it later.

        **Solution: Challenge-response with nonce**

        ```python
        class ReplayProtection:
            """Prevent replay attacks using nonces and timestamps"""

            def __init__(self):
                self.redis = redis.Redis()
                self.nonce_ttl = 300  # 5 minutes

            def generate_nonce(self, lock_id: str) -> str:
                """Generate unique nonce for lock operation"""
                nonce = secrets.token_hex(16)

                # Store nonce in Redis (prevent reuse)
                key = f"nonce:{lock_id}:{nonce}"
                self.redis.setex(key, self.nonce_ttl, "1")

                return nonce

            def verify_nonce(self, lock_id: str, nonce: str) -> bool:
                """Verify nonce hasn't been used"""
                key = f"nonce:{lock_id}:{nonce}"

                # Check if exists
                if not self.redis.exists(key):
                    return False  # Already used or expired

                # Delete nonce (one-time use)
                self.redis.delete(key)

                return True

            def verify_timestamp(self, timestamp_ms: int, tolerance_ms: int = 30000) -> bool:
                """
                Verify timestamp is recent (within tolerance)

                Args:
                    timestamp_ms: Command timestamp
                    tolerance_ms: Acceptable time difference (default 30s)

                Returns:
                    True if timestamp is recent
                """
                now_ms = int(time.time() * 1000)
                diff = abs(now_ms - timestamp_ms)

                return diff <= tolerance_ms
        ```

        **Lock firmware implementation:**

        ```c
        // Lock firmware nonce verification
        bool verify_command(CommandPacket* cmd) {
            // 1. Verify timestamp (prevent old replays)
            uint64_t now_ms = get_system_time_ms();
            uint64_t timestamp_ms = cmd->timestamp;

            if (abs(now_ms - timestamp_ms) > 30000) {
                return false;  // Timestamp too old/future
            }

            // 2. Verify nonce (prevent immediate replays)
            if (nonce_cache_contains(cmd->nonce)) {
                return false;  // Nonce already used
            }

            // 3. Verify HMAC signature
            uint8_t expected_sig[32];
            hmac_sha256(cmd->message, cmd->message_len, shared_secret, 32, expected_sig);

            if (memcmp(cmd->signature, expected_sig, 32) != 0) {
                return false;  // Invalid signature
            }

            // Add nonce to cache (prevent reuse)
            nonce_cache_add(cmd->nonce);

            return true;
        }
        ```

        ---

        ## Key Rotation

        **Periodic key rotation:**

        ```python
        class KeyRotationManager:
            """Rotate encryption keys periodically"""

            def rotate_lock_master_key(self, lock_id: str):
                """
                Rotate lock master key (every 90 days)

                Process:
                1. Generate new master key
                2. Re-encrypt all virtual keys with new master
                3. Push new key to lock firmware (OTA update)
                4. Deactivate old key after grace period
                """
                # Generate new master key
                new_master_key = os.urandom(32)

                # Get all virtual keys for lock
                virtual_keys = db.query(
                    "SELECT * FROM virtual_keys WHERE lock_id = %s AND status = 'active'",
                    (lock_id,)
                )

                # Re-encrypt each virtual key
                old_master_key = self._get_lock_master_key(lock_id)

                for vkey in virtual_keys:
                    # Decrypt with old key
                    encrypted_data = vkey['encrypted_data']
                    plaintext = self._decrypt_aes_gcm(encrypted_data, old_master_key)

                    # Re-encrypt with new key
                    new_encrypted = self._encrypt_aes_gcm(plaintext, new_master_key)

                    # Update database
                    db.execute(
                        "UPDATE virtual_keys SET encrypted_data = %s WHERE key_id = %s",
                        (new_encrypted, vkey['key_id'])
                    )

                # Store new master key
                self._store_lock_master_key(lock_id, new_master_key)

                # Push to lock firmware via OTA
                firmware_service.push_key_update(lock_id, new_master_key)

                # Mark old key for deletion (grace period: 7 days)
                self._schedule_key_deletion(lock_id, old_master_key, days=7)
        ```

    === "🔋 Battery Optimization"

        ## The Challenge

        **Problem:** Maximize battery life (target: 6-12 months) on 4x AA batteries.

        **Power consumption breakdown:**

        | Component | Active Current | Duty Cycle | Average |
        |-----------|---------------|------------|---------|
        | MCU (Nordic nRF52) | 10 mA | 5% | 500 µA |
        | BLE advertising | 15 mA | 1% | 150 µA |
        | BLE connected | 10 mA | 0.1% | 10 µA |
        | Motor (lock/unlock) | 500 mA | 0.002% | 10 µA |
        | Door sensor (Hall) | 20 µA | 100% | 20 µA |
        | RTC (timekeeping) | 2 µA | 100% | 2 µA |
        | Sleep mode | 5 µA | ~94% | 5 µA |
        | **Total** | | | **~700 µA** |

        **Battery life:** 1000 mAh / 0.7 mA ≈ 1400 hours ≈ 58 days (2 months) - **needs optimization**

        ---

        ## Optimization Strategies

        ### 1. Reduce BLE Advertising Frequency

        ```c
        // Dynamic advertising based on context

        typedef enum {
            ADV_MODE_FAST,      // 100ms interval (near user)
            ADV_MODE_NORMAL,    // 1000ms interval (default)
            ADV_MODE_SLOW,      // 5000ms interval (idle)
        } AdvertisingMode;

        void set_advertising_mode(AdvertisingMode mode) {
            uint32_t interval_ms;

            switch (mode) {
                case ADV_MODE_FAST:
                    interval_ms = 100;  // Near user (detected by accelerometer)
                    break;
                case ADV_MODE_NORMAL:
                    interval_ms = 1000;  // Normal
                    break;
                case ADV_MODE_SLOW:
                    interval_ms = 5000;  // Idle (no motion)
                    break;
            }

            ble_gap_adv_params_t params = {
                .interval = interval_ms,
                .tx_power = -8  // Reduced power
            };

            sd_ble_gap_adv_start(&params);
        }

        // Accelerometer interrupt triggers fast mode
        void on_motion_detected() {
            set_advertising_mode(ADV_MODE_FAST);
            start_timer(FAST_MODE_TIMEOUT_MS);  // Return to normal after 30s
        }
        ```

        **Impact:** Reduce advertising current from 150 µA to 50 µA → **saves 100 µA**

        ---

        ### 2. Efficient Motor Control

        ```c
        // Optimize motor power usage

        void unlock_motor_optimized() {
            // 1. Use PWM to reduce peak current
            uint8_t pwm_duty = 80;  // 80% duty cycle (slower but less current)

            pwm_set_duty(MOTOR_PWM_CHANNEL, pwm_duty);
            enable_motor();

            // 2. Monitor position with encoder (stop when unlocked)
            while (get_lock_position() < UNLOCK_POSITION) {
                nrf_delay_ms(10);
            }

            disable_motor();

            // 3. Apply motor brake (prevent drift)
            set_motor_brake();

            // Battery savings: 500mA → 400mA, 1s → 0.8s
            // Average: 500mA × 1s → 400mA × 0.8s (20% reduction)
        }
        ```

        ---

        ### 3. Deep Sleep Mode

        ```c
        // Aggressive power management

        void enter_deep_sleep() {
            // Disable unused peripherals
            nrf_uarte_disable(UART_INST);
            nrf_spim_disable(SPI_INST);

            // Configure wake sources
            nrf_gpio_cfg_sense_input(DOOR_SENSOR_PIN, NRF_GPIO_PIN_NOPULL, NRF_GPIO_PIN_SENSE_LOW);
            nrf_gpio_cfg_sense_input(BUTTON_PIN, NRF_GPIO_PIN_PULLUP, NRF_GPIO_PIN_SENSE_LOW);

            // Enter System ON sleep (5 µA)
            sd_power_system_off();  // Wakes on GPIO or BLE event
        }

        // Wake-up sources:
        // 1. Door sensor (Hall effect) - detect door open/close
        // 2. BLE connection request
        // 3. Manual button press
        // 4. RTC timer (periodic sync)
        ```

        **Impact:** Reduce MCU current from 500 µA to 100 µA → **saves 400 µA**

        ---

        ### 4. Efficient Crypto Operations

        ```c
        // Use hardware crypto accelerator (nRF52 CCM)

        void encrypt_command_hw(uint8_t* plaintext, uint8_t* key, uint8_t* ciphertext) {
            // Configure CCM (AES-128-CCM in hardware)
            NRF_CCM->MODE = CCM_MODE_MODE_Encryption;
            NRF_CCM->CNFPTR = (uint32_t)key;
            NRF_CCM->INPTR = (uint32_t)plaintext;
            NRF_CCM->OUTPTR = (uint32_t)ciphertext;

            // Start encryption (hardware accelerated)
            NRF_CCM->TASKS_START = 1;

            // Wait for completion (< 1ms)
            while (!NRF_CCM->EVENTS_END);

            // Hardware crypto: 10x faster, 5x lower power than software
        }
        ```

        **Impact:** Reduce crypto overhead from 50 µA to 10 µA → **saves 40 µA**

        ---

        ## Optimized Battery Life

        **After optimizations:**

        | Component | Optimized Average |
        |-----------|------------------|
        | MCU | 100 µA |
        | BLE advertising | 50 µA |
        | BLE connected | 10 µA |
        | Motor | 10 µA |
        | Door sensor | 20 µA |
        | RTC | 2 µA |
        | Sleep mode | 5 µA |
        | Crypto | 10 µA |
        | **Total** | **~200 µA** |

        **Battery life:** 1000 mAh / 0.2 mA = 5000 hours ≈ 208 days ≈ **7 months**

        **With 1500 mAh battery pack:** 7500 hours ≈ **10.3 months** ✅

        ---

        ## Battery Monitoring

        ```python
        class BatteryMonitor:
            """Monitor and predict battery life"""

            VOLTAGE_FRESH = 6.0  # 4x 1.5V AA batteries
            VOLTAGE_DEPLETED = 4.4  # 4x 1.1V (end of life)

            def get_battery_level(self, voltage: float) -> int:
                """
                Calculate battery percentage from voltage

                Returns:
                    Battery level (0-100%)
                """
                if voltage >= self.VOLTAGE_FRESH:
                    return 100
                if voltage <= self.VOLTAGE_DEPLETED:
                    return 0

                # Linear approximation (good enough for alkaline)
                percentage = int(
                    (voltage - self.VOLTAGE_DEPLETED) /
                    (self.VOLTAGE_FRESH - self.VOLTAGE_DEPLETED) * 100
                )

                return max(0, min(100, percentage))

            def predict_days_remaining(
                self,
                current_voltage: float,
                usage_rate: float
            ) -> int:
                """
                Predict days until battery depletion

                Args:
                    current_voltage: Current battery voltage
                    usage_rate: Average operations per day

                Returns:
                    Estimated days remaining
                """
                current_percentage = self.get_battery_level(current_voltage)

                # Estimate based on usage rate
                # Baseline: 10 ops/day → 7 months
                baseline_lifetime_days = 210
                baseline_ops_per_day = 10

                adjusted_lifetime = baseline_lifetime_days * (baseline_ops_per_day / usage_rate)
                days_remaining = adjusted_lifetime * (current_percentage / 100)

                return int(days_remaining)

            def should_send_low_battery_alert(self, voltage: float) -> bool:
                """Send alert when battery below 20%"""
                return self.get_battery_level(voltage) <= 20
        ```

    === "🌐 Offline Operation"

        ## The Challenge

        **Problem:** Lock must operate reliably without internet connectivity.

        **Requirements:**
        - 100% Bluetooth functionality offline
        - Virtual key validation without cloud
        - Access logging with delayed sync
        - Time-based permissions without NTP

        ---

        ## Offline Architecture

        **Key design principles:**

        1. **Embedded key storage** - Virtual keys stored in lock flash memory
        2. **Local validation** - Cryptographic verification without server
        3. **Buffered logging** - Store logs locally, sync when online
        4. **RTC time sync** - Maintain accurate time with periodic NTP sync

        ---

        ## Virtual Key Synchronization

        **Sync protocol (when phone in Bluetooth range):**

        ```python
        class OfflineKeySync:
            """Synchronize virtual keys for offline operation"""

            async def sync_keys_to_lock(self, lock_id: str, mobile_app: BLEClient):
                """
                Push virtual keys from cloud to lock via Bluetooth

                Process:
                1. Get all active keys from database
                2. Encrypt keys for transmission
                3. Send to lock via BLE
                4. Lock stores in flash memory
                """
                # Get active keys for lock
                active_keys = db.query("""
                    SELECT key_id, recipient_user_id, permissions,
                           start_time, end_time, days_of_week, time_windows,
                           access_limit, secret
                    FROM virtual_keys
                    WHERE lock_id = %s AND status = 'active'
                """, (lock_id,))

                # Prepare key bundle
                key_bundle = []
                for key in active_keys:
                    key_bundle.append({
                        'key_id': key['key_id'],
                        'user_id': key['recipient_user_id'],
                        'secret': key['secret'],
                        'permissions': key['permissions'],
                        'schedule': {
                            'start': key['start_time'].timestamp() if key['start_time'] else None,
                            'end': key['end_time'].timestamp() if key['end_time'] else None,
                            'days': key['days_of_week'],
                            'windows': key['time_windows']
                        },
                        'limit': key['access_limit']
                    })

                # Encrypt bundle
                encrypted_bundle = security_manager.encrypt_key_bundle(key_bundle, lock_id)

                # Send to lock via BLE (KEY_SYNC characteristic)
                await mobile_app.write_gatt_char(
                    SmartLockBLEClient.KEY_SYNC_UUID,
                    encrypted_bundle
                )

                # Lock responds with confirmation
                response = await mobile_app.read_gatt_char(
                    SmartLockBLEClient.KEY_SYNC_UUID
                )

                sync_status = self._parse_sync_response(response)
                return sync_status  # {status: 'success', keys_synced: 5}
        ```

        **Lock firmware (key storage):**

        ```c
        // Lock firmware key storage (flash memory)

        #define MAX_VIRTUAL_KEYS 20
        #define FLASH_PAGE_KEYS 0x7F000  // Flash page for key storage

        typedef struct {
            uint8_t key_id[16];
            uint8_t user_id[16];
            uint8_t secret[32];

            // Permissions
            bool can_lock;
            bool can_unlock;

            // Schedule
            uint64_t start_timestamp;
            uint64_t end_timestamp;
            uint8_t days_of_week;  // Bitmap: 0x01=Sunday, 0x02=Monday, etc.
            uint16_t time_windows[4][2];  // Up to 4 time windows (start, end in minutes)

            // Usage
            uint16_t access_limit;
            uint16_t access_count;

            bool active;
        } VirtualKey;

        VirtualKey g_virtual_keys[MAX_VIRTUAL_KEYS];
        uint8_t g_key_count = 0;

        void sync_virtual_keys(uint8_t* encrypted_bundle, uint16_t len) {
            // Decrypt bundle
            uint8_t plaintext[1024];
            decrypt_aes_gcm(encrypted_bundle, len, lock_master_key, plaintext);

            // Parse keys from JSON
            parse_key_bundle(plaintext, g_virtual_keys, &g_key_count);

            // Write to flash (persistent storage)
            nrf_nvmc_page_erase(FLASH_PAGE_KEYS);
            nrf_nvmc_write_words(FLASH_PAGE_KEYS, (uint32_t*)g_virtual_keys,
                                 sizeof(VirtualKey) * g_key_count);

            // Send confirmation
            send_sync_response(g_key_count);
        }

        bool validate_key_offline(VirtualKey* key, uint64_t current_timestamp) {
            // Check if key active
            if (!key->active) return false;

            // Check time range
            if (key->start_timestamp > 0 && current_timestamp < key->start_timestamp)
                return false;
            if (key->end_timestamp > 0 && current_timestamp > key->end_timestamp)
                return false;

            // Check day of week
            time_t now = current_timestamp / 1000;
            struct tm* tm_info = localtime(&now);
            uint8_t day_bit = 1 << tm_info->tm_wday;
            if ((key->days_of_week & day_bit) == 0)
                return false;

            // Check time window
            uint16_t current_minutes = tm_info->tm_hour * 60 + tm_info->tm_min;
            bool in_window = false;
            for (int i = 0; i < 4; i++) {
                uint16_t start = key->time_windows[i][0];
                uint16_t end = key->time_windows[i][1];
                if (start == 0 && end == 0) continue;  // No window
                if (current_minutes >= start && current_minutes <= end) {
                    in_window = true;
                    break;
                }
            }
            if (!in_window) return false;

            // Check access limit
            if (key->access_limit > 0 && key->access_count >= key->access_limit)
                return false;

            return true;
        }
        ```

        ---

        ## Offline Access Logging

        **Buffered logging:**

        ```c
        // Lock firmware - store logs locally

        #define MAX_BUFFERED_LOGS 500
        #define FLASH_PAGE_LOGS 0x7E000

        typedef struct {
            uint64_t timestamp;
            uint8_t user_id[16];
            uint8_t key_id[16];
            uint8_t action;  // 0x10=lock, 0x11=unlock
            uint8_t status;  // 0x00=success, 0x01=denied
            uint8_t battery_level;
        } AccessLog;

        AccessLog g_log_buffer[MAX_BUFFERED_LOGS];
        uint16_t g_log_count = 0;

        void log_access_offline(AccessLog* log) {
            // Add to buffer
            if (g_log_count < MAX_BUFFERED_LOGS) {
                memcpy(&g_log_buffer[g_log_count], log, sizeof(AccessLog));
                g_log_count++;

                // Write to flash periodically (every 10 logs)
                if (g_log_count % 10 == 0) {
                    flush_logs_to_flash();
                }
            }
        }

        void sync_logs_to_cloud() {
            // When WiFi/Bluetooth connection available
            for (uint16_t i = 0; i < g_log_count; i++) {
                send_log_to_cloud(&g_log_buffer[i]);
            }

            // Clear buffer after successful sync
            g_log_count = 0;
            nrf_nvmc_page_erase(FLASH_PAGE_LOGS);
        }
        ```

        **Cloud sync service:**

        ```python
        class OfflineLogSync:
            """Sync access logs from lock to cloud"""

            async def sync_logs(self, lock_id: str, logs: List[dict]):
                """
                Receive buffered logs from lock and store in database

                Args:
                    lock_id: Lock identifier
                    logs: List of access log entries
                """
                # Batch insert to Cassandra
                batch = []
                for log in logs:
                    batch.append({
                        'lock_id': lock_id,
                        'log_time': datetime.fromtimestamp(log['timestamp'] / 1000),
                        'log_id': str(uuid.uuid4()),
                        'user_id': log['user_id'],
                        'key_id': log.get('key_id'),
                        'action': ['lock', 'unlock'][log['action'] - 0x10],
                        'status': ['success', 'denied'][log['status']],
                        'battery_level': log['battery_level'],
                        'method': 'bluetooth'  # Offline = Bluetooth
                    })

                # Insert in parallel
                await self._batch_insert_cassandra(batch)

                # Trigger notifications for important events
                await self._process_log_events(lock_id, logs)

            async def _process_log_events(self, lock_id: str, logs: List[dict]):
                """Send notifications for important events"""
                for log in logs:
                    # Notify owner of failed access attempts
                    if log['status'] != 0x00:  # Failed
                        await notification_service.send_alert(
                            lock_id,
                            f"Failed unlock attempt at {log['timestamp']}"
                        )
        ```

        ---

        ## Time Synchronization

        **Problem:** Lock needs accurate time for schedule validation without internet.

        **Solution: RTC with periodic NTP sync**

        ```python
        class LockTimeSync:
            """Maintain accurate time on lock device"""

            async def sync_time_to_lock(self, lock_id: str, mobile_app: BLEClient):
                """
                Sync current time to lock via Bluetooth

                Called:
                - When mobile app connects
                - Periodically (every 24 hours)
                - After firmware update
                """
                # Get accurate time from NTP
                ntp_time = self._get_ntp_time()

                # Send to lock
                time_packet = struct.pack('!Q', int(ntp_time * 1000))  # Unix ms
                await mobile_app.write_gatt_char(
                    SmartLockBLEClient.TIME_SYNC_UUID,
                    time_packet
                )

            def _get_ntp_time(self) -> float:
                """Get time from NTP server"""
                import ntplib
                client = ntplib.NTPClient()
                response = client.request('pool.ntp.org', version=3)
                return response.tx_time
        ```

        **Lock firmware (RTC):**

        ```c
        // Nordic nRF52 RTC (Real-Time Counter)

        void init_rtc() {
            // Configure RTC (32.768 kHz crystal)
            NRF_RTC1->PRESCALER = 0;  // 1 tick = 30.5 µs
            NRF_RTC1->TASKS_START = 1;
        }

        void set_time_from_ntp(uint64_t unix_ms) {
            // Set system time
            g_system_time_offset = unix_ms - (NRF_RTC1->COUNTER * 305 / 10000);
        }

        uint64_t get_current_time_ms() {
            // Calculate current time
            uint64_t rtc_ms = NRF_RTC1->COUNTER * 305 / 10000;
            return g_system_time_offset + rtc_ms;
        }

        // Time drift: ~20 ppm (±1.7 seconds/day)
        // Sync every 24 hours keeps error < 2 seconds (acceptable)
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 100K locks to 10M locks.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Key validation** | 🟡 Moderate | Redis caching, read replicas |
    | **Access log writes** | ✅ Yes | Cassandra cluster (scale horizontally) |
    | **Push notifications** | 🟡 Moderate | FCM/APNs (managed service) |
    | **Lock state queries** | 🟢 No | Redis cache (< 10ms) |

    ---

    ## Horizontal Scaling

    ### Database Sharding

    ```python
    class LockSharding:
        """Shard locks across PostgreSQL instances"""

        def __init__(self):
            self.shard_count = 16
            self.shards = [
                PostgreSQLConnection(f"lock-db-{i}.example.com")
                for i in range(self.shard_count)
            ]

        def get_shard(self, lock_id: str) -> PostgreSQLConnection:
            """Consistent hashing to determine shard"""
            shard_index = int(lock_id.split('-')[0], 16) % self.shard_count
            return self.shards[shard_index]

        def query_lock(self, lock_id: str) -> dict:
            """Query lock from appropriate shard"""
            shard = self.get_shard(lock_id)
            return shard.query("SELECT * FROM locks WHERE lock_id = %s", (lock_id,))
    ```

    ---

    ## Caching Strategy

    ```python
    class LockCacheManager:
        """Multi-level caching for lock operations"""

        def __init__(self):
            self.redis = redis.Redis()
            self.local_cache = {}  # In-memory L1 cache
            self.ttl = 300  # 5 minutes

        def get_virtual_key(self, key_id: str) -> dict:
            """
            Get virtual key with multi-level caching

            L1: In-memory (local process)
            L2: Redis (shared)
            L3: PostgreSQL (database)
            """
            # L1: Local cache (fastest, ~1µs)
            if key_id in self.local_cache:
                return self.local_cache[key_id]

            # L2: Redis cache (~1ms)
            cached = self.redis.get(f"vkey:{key_id}")
            if cached:
                key_data = json.loads(cached)
                self.local_cache[key_id] = key_data
                return key_data

            # L3: Database (~10ms)
            key_data = db.query(
                "SELECT * FROM virtual_keys WHERE key_id = %s",
                (key_id,)
            )

            # Populate caches
            self.redis.setex(f"vkey:{key_id}", self.ttl, json.dumps(key_data))
            self.local_cache[key_id] = key_data

            return key_data
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 10M locks:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $43,200 (200 instances @ $6/day) |
    | **RDS PostgreSQL** | $17,280 (16 shards @ $1,080/month) |
    | **Cassandra (logs)** | $21,600 (20 nodes @ $1,080/month) |
    | **Redis Cache** | $8,640 (10 nodes @ $864/month) |
    | **S3 (backups)** | $2,300 (100 TB @ $23/TB) |
    | **CloudWatch** | $1,200 |
    | **Total** | **$94,220/month** |

    **Per-lock cost:** $94,220 / 10M = **$0.0094/month = $0.11/year per lock**

    **Revenue:** Assume $10/month subscription. 10M locks = $100M/month. Infrastructure is 0.09% of revenue.

    ---

    ## Monitoring & Alerting

    ```python
    class LockSystemMonitoring:
        """Monitor system health and performance"""

        def __init__(self):
            self.cloudwatch = boto3.client('cloudwatch')

        def track_lock_operation(self, lock_id: str, operation: str, latency_ms: float, success: bool):
            """Track lock operation metrics"""
            self.cloudwatch.put_metric_data(
                Namespace='SmartLock',
                MetricData=[
                    {
                        'MetricName': 'OperationLatency',
                        'Value': latency_ms,
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {'Name': 'Operation', 'Value': operation},
                            {'Name': 'Success', 'Value': str(success)}
                        ]
                    },
                    {
                        'MetricName': 'OperationCount',
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Operation', 'Value': operation},
                            {'Name': 'Status', 'Value': 'success' if success else 'failure'}
                        ]
                    }
                ]
            )

        def create_alarms(self):
            """Create CloudWatch alarms"""
            # High latency alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName='SmartLock-HighLatency',
                MetricName='OperationLatency',
                Namespace='SmartLock',
                Statistic='Average',
                Period=300,
                EvaluationPeriods=2,
                Threshold=2000,  # 2 seconds
                ComparisonOperator='GreaterThanThreshold',
                AlarmActions=['arn:aws:sns:us-east-1:123456789:alerts']
            )

            # High failure rate alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName='SmartLock-HighFailureRate',
                MetricName='OperationCount',
                Namespace='SmartLock',
                Statistic='Sum',
                Period=300,
                EvaluationPeriods=2,
                Threshold=100,  # 100 failures in 5 minutes
                ComparisonOperator='GreaterThanThreshold',
                Dimensions=[{'Name': 'Status', 'Value': 'failure'}],
                AlarmActions=['arn:aws:sns:us-east-1:123456789:alerts']
            )
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Bluetooth-first architecture** - Offline reliability with local crypto verification
    2. **Dual protocol** - BLE for local, WiFi bridge for remote convenience
    3. **End-to-end encryption** - AWS KMS + AES-256-GCM for all communications
    4. **Virtual key synchronization** - Push keys to lock flash for offline validation
    5. **Battery optimization** - Aggressive sleep modes, hardware crypto, adaptive BLE
    6. **Buffered logging** - Local storage with delayed cloud sync
    7. **Time-based permissions** - RTC with periodic NTP sync for schedule enforcement

    ---

    ## Interview Tips

    ✅ **Start with offline requirement** - Emphasize Bluetooth-first design

    ✅ **Discuss security deeply** - Encryption, key management, replay protection

    ✅ **Battery life critical** - Show specific optimizations and calculations

    ✅ **Virtual key sharing** - Schedule enforcement, access limits, revocation

    ✅ **Access audit trail** - Complete logging for security compliance

    ✅ **Remote access** - WiFi bridge for convenience (trade-off discussion)

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"What if phone battery dies?"** | Lock has physical key backup, also supports PIN code entry on keypad (if equipped) |
    | **"How to handle lost phone?"** | Revoke virtual keys via web portal, lock owner can always access via their account |
    | **"Bluetooth pairing security?"** | BLE Secure Connections (ECDH), only pair when button pressed, shared secret never transmitted |
    | **"What if lock firmware has bug?"** | OTA firmware updates via mobile app or WiFi bridge, rollback capability, staged rollouts |
    | **"How to prevent relay attacks?"** | Proximity verification (RSSI), nonce + timestamp, optional geofencing |
    | **"Scale to 100M locks?"** | Shard databases by lock_id, Cassandra for logs, Redis cache, stateless API servers |
    | **"Fire safety (auto-unlock)?"** | Smoke detector integration via smart home APIs, configurable emergency unlock |
    | **"Multi-lock support?"** | User can have keys to multiple locks, mobile app shows all accessible locks |

    ---

    ## Real-World Examples

    ### August Smart Lock

    - **Architecture:** Bluetooth + WiFi bridge (August Connect)
    - **Battery:** 4x AA, 6-12 months
    - **Features:** Auto-lock, geofencing, guest keys, DoorSense (door open sensor)
    - **Integration:** Alexa, Google Home, HomeKit

    ### Amazon Key

    - **Architecture:** Cloud-first with WiFi camera integration
    - **Use case:** Package delivery, in-home services
    - **Security:** Real-time notifications, video verification
    - **Scale:** Used by Amazon delivery drivers

    ### Yale Assure Lock

    - **Architecture:** Z-Wave/Zigbee for smart home integration
    - **Battery:** 9V or 4x AA
    - **Features:** Touchscreen keypad, 25 PIN codes
    - **Offline:** Fully functional without internet

    ---

    ## Security Best Practices

    1. **Encryption everywhere** - E2E encryption for all communications
    2. **Key rotation** - Rotate master keys every 90 days
    3. **Audit logging** - Log all operations (success and failures)
    4. **Rate limiting** - Prevent brute-force PIN/key attacks
    5. **Tamper detection** - Accelerometer detects physical attacks
    6. **Secure boot** - Verify firmware signature on startup
    7. **Intrusion detection** - Alert on multiple failed unlock attempts

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Amazon Key, August, Yale, Schlage, Kwikset

---

*Master this problem and you'll be ready for: Smart home systems, IoT platforms, access control systems*
