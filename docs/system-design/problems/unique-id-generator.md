# Design Unique ID Generator (Snowflake)

A distributed system that generates globally unique, sortable, 64-bit identifiers at high throughput without coordination between nodes.

**Difficulty:** 🟢 Easy | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 30-45 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1M IDs/sec per generator, 1000+ distributed nodes, billions of IDs per day |
| **Key Challenges** | Uniqueness guarantee, time-ordering, no coordination, high performance, no single point of failure |
| **Core Concepts** | Snowflake algorithm, timestamp + machine ID + sequence, 64-bit IDs, distributed generation |
| **Companies** | Twitter (Snowflake), Instagram, Discord, MongoDB (ObjectID), Snowflake DB |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Generate Unique ID** | Generate globally unique 64-bit integer IDs | P0 (Must have) |
    | **Time-Ordered** | IDs should be sortable by creation time | P0 (Must have) |
    | **High Performance** | < 1ms generation latency, 1M IDs/sec per node | P0 (Must have) |
    | **Distributed** | Multiple nodes generate IDs independently | P0 (Must have) |
    | **No Coordination** | No inter-node communication required | P0 (Must have) |
    | **Decode Timestamp** | Extract creation timestamp from ID | P1 (Should have) |
    | **Human Readable** | Optional: URL-safe string representation | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Sequential IDs (reveals business metrics)
    - Cryptographically secure IDs
    - IDs shorter than 64 bits
    - Custom alphabet encoding
    - Batch ID generation API

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 1ms p99 | Must not block application logic |
    | **Throughput** | 1M IDs/sec per node | Support high-traffic systems |
    | **Availability** | 99.99% uptime | Critical for all write operations |
    | **Uniqueness** | 100% guaranteed | Collisions cause data corruption |
    | **Sortability** | Time-ordered (1ms resolution) | Enable efficient range queries |
    | **Scalability** | Support 1000+ nodes | Horizontal scaling without coordination |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total system capacity:
    - Number of generator nodes: 1000
    - IDs per second per node: 1M
    - Total capacity: 1000 × 1M = 1B IDs/sec

    Realistic usage (e.g., Twitter-scale):
    - New tweets: 6,000 IDs/sec
    - New users: 500 IDs/sec
    - Media uploads: 2,000 IDs/sec
    - Likes/retweets: 100,000 IDs/sec
    - Total: ~110K IDs/sec (well within capacity)

    Peak traffic (10x normal):
    - Peak IDs/sec: 1.1M IDs/sec
    - Required nodes: 2 nodes (with headroom)
    - Actual deployment: 10+ nodes for redundancy
    ```

    ### Storage Estimates

    ```
    ID storage:
    - ID size: 8 bytes (64-bit integer)
    - IDs per day: 110K × 86,400 = 9.5B IDs/day
    - Daily storage: 9.5B × 8 bytes = 76 GB/day

    For 10 years:
    - Total IDs: 9.5B × 365 × 10 = 34.7 trillion IDs
    - Storage: 34.7T × 8 bytes = 278 TB (IDs only)

    Note: IDs are typically stored as part of other records,
    so this is included in application database size.
    ```

    ### Memory Estimates

    ```
    ID generator state:
    - Last timestamp: 8 bytes
    - Sequence counter: 4 bytes
    - Machine ID: 4 bytes
    - Total per node: 16 bytes (negligible)

    Cache recent IDs (for collision detection):
    - Last 1M IDs: 1M × 8 bytes = 8 MB per node
    - 1000 nodes: 8 GB total (negligible)
    ```

    ### Throughput Estimates

    ```
    Per node:
    - 1M IDs/sec × 8 bytes = 8 MB/sec per node
    - Negligible network bandwidth

    System-wide:
    - 110K IDs/sec × 8 bytes = 880 KB/sec
    - Total: < 10 Mbps (trivial)
    ```

    ---

    ## Key Assumptions

    1. 64-bit integer IDs are acceptable (not 128-bit UUIDs)
    2. 1ms timestamp resolution is sufficient
    3. IDs must be time-ordered (within same node)
    4. Each node has unique machine ID (0-1023)
    5. Clock drift < 1 second between nodes
    6. Node failure is independent (no cascade)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **No coordination:** Each node generates IDs independently
    2. **64-bit structure:** Timestamp + Machine ID + Sequence
    3. **Time-ordered:** Higher timestamp = larger ID
    4. **High availability:** No single point of failure
    5. **Simple design:** Minimal state, fast generation

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Application Layer"
            App1[App Server 1]
            App2[App Server 2]
            App3[App Server 3]
            AppN[App Server N]
        end

        subgraph "ID Generator Layer"
            Gen1[ID Generator 1<br/>Machine ID: 001<br/>1M IDs/sec]
            Gen2[ID Generator 2<br/>Machine ID: 002<br/>1M IDs/sec]
            Gen3[ID Generator 3<br/>Machine ID: 003<br/>1M IDs/sec]
            GenN[ID Generator N<br/>Machine ID: 1000<br/>1M IDs/sec]
        end

        subgraph "Configuration"
            Config[Config Service<br/>ZooKeeper<br/>Machine ID registry]
            NTP[NTP Server<br/>Time synchronization]
        end

        subgraph "Monitoring"
            Metrics[Metrics<br/>Prometheus]
            Alerts[Alerting<br/>PagerDuty]
        end

        App1 --> Gen1
        App2 --> Gen2
        App3 --> Gen3
        AppN --> GenN

        Gen1 -.->|Register| Config
        Gen2 -.->|Register| Config
        Gen3 -.->|Register| Config
        GenN -.->|Register| Config

        Gen1 -.->|Sync time| NTP
        Gen2 -.->|Sync time| NTP
        Gen3 -.->|Sync time| NTP
        GenN -.->|Sync time| NTP

        Gen1 --> Metrics
        Gen2 --> Metrics
        Gen3 --> Metrics
        GenN --> Metrics

        Metrics --> Alerts

        style Gen1 fill:#e1f5ff
        style Gen2 fill:#e1f5ff
        style Gen3 fill:#e1f5ff
        style GenN fill:#e1f5ff
        style Config fill:#fff4e1
        style NTP fill:#e8f5e9
        style Metrics fill:#f3e5f5
    ```

    ---

    ## Snowflake ID Structure (64 bits)

    ```
    Twitter Snowflake (64 bits):

    +--------------------------------------------------------------------------+
    | 1 bit    | 41 bits              | 10 bits      | 12 bits               |
    | (unused) | Timestamp (ms)       | Machine ID   | Sequence Number       |
    | 0        | Milliseconds         | 0-1023       | 0-4095                |
    +--------------------------------------------------------------------------+

    Bit allocation breakdown:

    - Bit 0 (1 bit): Reserved/Unused (always 0)
      → Ensures ID is always positive

    - Bits 1-41 (41 bits): Timestamp in milliseconds
      → Epoch: Custom epoch (e.g., Jan 1, 2020 00:00:00 UTC)
      → Range: 2^41 ms = 69 years from epoch
      → Example: If epoch is 2020, IDs valid until 2089

    - Bits 42-51 (10 bits): Machine/Datacenter ID
      → Range: 0 to 1023 (2^10 = 1024 machines)
      → Can split: 5 bits datacenter (32 DCs) + 5 bits machine (32 per DC)
      → Uniquely identifies generator node

    - Bits 52-63 (12 bits): Sequence number
      → Range: 0 to 4095 (2^12 = 4096 IDs per millisecond per machine)
      → Resets to 0 every millisecond
      → Allows 4.096M IDs/sec per node
    ```

    **Example ID Generation:**

    ```
    Timestamp: 1643723400000 (Jan 29, 2026 10:30:00 UTC)
    Machine ID: 123
    Sequence: 456

    Binary:
    0 | 00000000101111110101110101010001011111000 | 0001111011 | 000111001000
    ↑   ↑                                           ↑            ↑
    Unused  Timestamp (41 bits)                     Machine ID   Sequence

    Decimal ID: 386848582584012328
    Hex: 0x55E5D5177BC8
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **In-process generation** | < 1ms latency, no network calls | External service (high latency, single point of failure) |
    | **64-bit integer** | Efficient storage/indexing, time-ordered | UUID (128-bit, not time-ordered), GUID (not sortable) |
    | **Machine ID** | Uniqueness across nodes, no coordination | Centralized ID service (bottleneck), random (collision risk) |
    | **Sequence counter** | Multiple IDs per millisecond | Wait for next ms (low throughput), random (collision risk) |
    | **ZooKeeper** | Machine ID registry, prevent duplicates | Manual config (error-prone), database (slower) |
    | **NTP** | Time synchronization across nodes | System clock (drift issues), GPS (expensive) |

    **Key Trade-off:** We chose **timestamp-based IDs over random UUIDs** for sortability and efficient database indexing, trading randomness for time-ordering.

    ---

    ## API Design

    ### 1. Generate ID

    **Request:**
    ```http
    GET /api/v1/id/generate
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": 386848582584012328,
      "id_str": "386848582584012328",
      "timestamp_ms": 1643723400000,
      "machine_id": 123,
      "sequence": 456,
      "created_at": "2026-01-29T10:30:00.000Z"
    }
    ```

    **Design Notes:**

    - Typically called as library function (not HTTP endpoint)
    - Sub-millisecond response time
    - No database access required
    - Thread-safe implementation

    ---

    ### 2. Batch Generate IDs

    **Request:**
    ```http
    POST /api/v1/id/generate/batch
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "count": 1000
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "ids": [
        386848582584012328,
        386848582584012329,
        386848582584012330,
        // ... 997 more
      ],
      "count": 1000,
      "generated_in_ms": 0.8
    }
    ```

    **Design Notes:**

    - Pre-allocate IDs for bulk operations
    - Still maintains time-ordering
    - Useful for batch inserts

    ---

    ### 3. Decode ID (Extract Metadata)

    **Request:**
    ```http
    GET /api/v1/id/decode/386848582584012328
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": 386848582584012328,
      "timestamp_ms": 1643723400000,
      "created_at": "2026-01-29T10:30:00.000Z",
      "machine_id": 123,
      "datacenter_id": 3,
      "worker_id": 27,
      "sequence": 456,
      "age_seconds": 3600
    }
    ```

    **Design Notes:**

    - Read-only operation (no state changes)
    - Useful for debugging and analytics
    - Can implement as pure function (no dependencies)

    ---

    ## Algorithm Implementation

    ### Snowflake ID Generator (Python)

    ```python
    import time
    import threading

    class SnowflakeGenerator:
        """
        Twitter Snowflake ID generator

        64-bit ID format:
        - 1 bit: unused (always 0)
        - 41 bits: timestamp (milliseconds since epoch)
        - 10 bits: machine ID (0-1023)
        - 12 bits: sequence (0-4095)
        """

        # Epoch: Jan 1, 2020 00:00:00 UTC
        EPOCH = 1577836800000

        # Bit lengths
        TIMESTAMP_BITS = 41
        MACHINE_ID_BITS = 10
        SEQUENCE_BITS = 12

        # Max values
        MAX_MACHINE_ID = (1 << MACHINE_ID_BITS) - 1  # 1023
        MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1      # 4095

        # Bit shifts
        TIMESTAMP_SHIFT = MACHINE_ID_BITS + SEQUENCE_BITS  # 22
        MACHINE_ID_SHIFT = SEQUENCE_BITS                    # 12

        def __init__(self, machine_id: int):
            """
            Initialize Snowflake generator

            Args:
                machine_id: Unique machine ID (0-1023)
            """
            if machine_id < 0 or machine_id > self.MAX_MACHINE_ID:
                raise ValueError(f"Machine ID must be between 0 and {self.MAX_MACHINE_ID}")

            self.machine_id = machine_id
            self.sequence = 0
            self.last_timestamp = -1
            self.lock = threading.Lock()

        def generate_id(self) -> int:
            """
            Generate unique 64-bit ID

            Returns:
                Globally unique 64-bit integer ID
            """
            with self.lock:
                timestamp = self._current_timestamp()

                # Clock moved backwards
                if timestamp < self.last_timestamp:
                    raise Exception(f"Clock moved backwards. Refusing to generate ID for {self.last_timestamp - timestamp}ms")

                # Same millisecond - increment sequence
                if timestamp == self.last_timestamp:
                    self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE

                    # Sequence overflow - wait for next millisecond
                    if self.sequence == 0:
                        timestamp = self._wait_next_millis(self.last_timestamp)
                else:
                    # New millisecond - reset sequence
                    self.sequence = 0

                self.last_timestamp = timestamp

                # Construct ID
                id = (
                    ((timestamp - self.EPOCH) << self.TIMESTAMP_SHIFT) |
                    (self.machine_id << self.MACHINE_ID_SHIFT) |
                    self.sequence
                )

                return id

        def decode_id(self, id: int) -> dict:
            """
            Decode Snowflake ID to extract components

            Args:
                id: 64-bit Snowflake ID

            Returns:
                Dictionary with timestamp, machine_id, sequence
            """
            # Extract components using bit masks and shifts
            sequence = id & self.MAX_SEQUENCE
            machine_id = (id >> self.MACHINE_ID_SHIFT) & self.MAX_MACHINE_ID
            timestamp = (id >> self.TIMESTAMP_SHIFT) + self.EPOCH

            return {
                'id': id,
                'timestamp_ms': timestamp,
                'created_at': self._timestamp_to_datetime(timestamp),
                'machine_id': machine_id,
                'sequence': sequence,
                'age_seconds': (self._current_timestamp() - timestamp) / 1000
            }

        def _current_timestamp(self) -> int:
            """Get current timestamp in milliseconds"""
            return int(time.time() * 1000)

        def _wait_next_millis(self, last_timestamp: int) -> int:
            """Wait until next millisecond"""
            timestamp = self._current_timestamp()
            while timestamp <= last_timestamp:
                timestamp = self._current_timestamp()
            return timestamp

        def _timestamp_to_datetime(self, timestamp: int) -> str:
            """Convert millisecond timestamp to ISO 8601 string"""
            from datetime import datetime
            return datetime.utcfromtimestamp(timestamp / 1000).isoformat() + 'Z'

    # Usage example
    generator = SnowflakeGenerator(machine_id=123)

    # Generate single ID
    id = generator.generate_id()
    print(f"Generated ID: {id}")

    # Decode ID
    info = generator.decode_id(id)
    print(f"ID components: {info}")

    # Generate multiple IDs
    ids = [generator.generate_id() for _ in range(10)]
    print(f"Generated {len(ids)} IDs: {ids}")
    ```

    ---

    ## Data Flow Diagram

    ### ID Generation Flow

    ```mermaid
    sequenceDiagram
        participant App as Application
        participant Gen as ID Generator
        participant Clock as System Clock
        participant ZK as ZooKeeper

        Note over Gen: Initialization
        Gen->>ZK: Register machine ID
        ZK-->>Gen: Machine ID: 123
        Gen->>Clock: Get current time
        Clock-->>Gen: 1643723400000 ms

        Note over App,Gen: ID Generation (< 1ms)
        App->>Gen: generate_id()
        Gen->>Gen: Lock mutex
        Gen->>Clock: Get timestamp
        Clock-->>Gen: timestamp_ms

        alt Same millisecond as last ID
            Gen->>Gen: Increment sequence (456 -> 457)
            alt Sequence overflow (> 4095)
                Gen->>Clock: Wait for next millisecond
                Clock-->>Gen: New timestamp
                Gen->>Gen: Reset sequence to 0
            end
        else New millisecond
            Gen->>Gen: Reset sequence to 0
        end

        Gen->>Gen: Construct ID:<br/>(timestamp << 22) | (machine_id << 12) | sequence
        Gen->>Gen: Update last_timestamp
        Gen->>Gen: Unlock mutex
        Gen-->>App: ID: 386848582584012328

        Note over App: Use ID for database insert
    ```

    **Flow Explanation:**

    1. **Initialization** - Register unique machine ID with ZooKeeper (one-time)
    2. **Get timestamp** - Query system clock (< 10µs)
    3. **Check last timestamp** - Compare with previous ID timestamp
    4. **Increment sequence** - If same millisecond, increment counter
    5. **Handle overflow** - If sequence > 4095, wait for next millisecond
    6. **Construct ID** - Bit-shift and combine components
    7. **Return immediately** - No database/network calls

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical aspects of ID generation.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Clock Synchronization** | How to handle clock drift between nodes? | NTP sync, clock skew detection, refuse IDs on backwards clock |
    | **High Availability** | How to ensure no single point of failure? | In-process generation, stateless nodes, ZooKeeper for coordination |
    | **Alternative Approaches** | What are other ID generation strategies? | UUID, MongoDB ObjectID, Database sequences, Composite keys |
    | **Clock Moving Backwards** | How to handle NTP corrections? | Refuse to generate, wait for time to catch up, use backup sequence |

    ---

    === "🕐 Clock Synchronization"

        ## The Challenge

        **Problem:** Distributed nodes have clock drift. If clocks differ by 1 second, IDs generated in past may collide with future IDs.

        **Example scenario:**

        ```
        Time = 10:00:00.000
        Node A generates ID: 123456789 (timestamp = 10:00:00.000)

        Time = 10:00:00.500
        Node B's clock is 2 seconds behind (shows 09:59:58.500)
        Node B generates ID: 123456700 (timestamp = 09:59:58.500)

        Problem: Node B's ID is SMALLER than Node A's earlier ID
        → Breaks time-ordering guarantee
        → Range queries may miss Node B's record
        ```

        ---

        ## NTP Synchronization

        **Network Time Protocol (NTP):**

        - Synchronizes clocks across network
        - Typical accuracy: ± 10ms on LAN, ± 100ms on WAN
        - Uses hierarchy: Stratum 0 (atomic clock) → Stratum 1 → Stratum 2 → ...

        **Implementation:**

        ```python
        import ntplib
        import time

        class ClockMonitor:
            """Monitor system clock and detect drift"""

            NTP_SERVERS = [
                'time.google.com',
                'time.cloudflare.com',
                'pool.ntp.org'
            ]

            MAX_CLOCK_DRIFT_MS = 100  # Alert if drift > 100ms

            def __init__(self):
                self.ntp_client = ntplib.NTPClient()
                self.last_check = 0
                self.CHECK_INTERVAL = 300  # Check every 5 minutes

            def check_clock_drift(self) -> tuple[bool, float]:
                """
                Check if system clock is synchronized with NTP

                Returns:
                    (is_synchronized, drift_ms)
                """
                now = time.time()

                # Rate limit checks
                if now - self.last_check < self.CHECK_INTERVAL:
                    return (True, 0)

                self.last_check = now

                # Try multiple NTP servers
                for ntp_server in self.NTP_SERVERS:
                    try:
                        response = self.ntp_client.request(ntp_server, timeout=2)
                        ntp_time = response.tx_time
                        system_time = time.time()

                        # Calculate drift
                        drift_ms = abs(ntp_time - system_time) * 1000

                        if drift_ms > self.MAX_CLOCK_DRIFT_MS:
                            logger.warning(f"Clock drift detected: {drift_ms:.2f}ms")
                            self._alert_clock_drift(drift_ms)
                            return (False, drift_ms)

                        return (True, drift_ms)

                    except Exception as e:
                        logger.error(f"NTP check failed for {ntp_server}: {e}")
                        continue

                # All NTP servers failed
                logger.error("All NTP servers unreachable")
                return (False, -1)

            def _alert_clock_drift(self, drift_ms: float):
                """Send alert for clock drift"""
                # Send to monitoring system (PagerDuty, Slack, etc.)
                alert_message = f"⚠️ Clock drift detected: {drift_ms:.2f}ms"
                # monitoring.send_alert(alert_message)
        ```

        ---

        ## Clock Moving Backwards

        **Problem:** NTP corrections can move clock backwards.

        **Example:**

        ```
        Time = 10:00:01.000
        Generator creates ID with timestamp = 10:00:01.000

        Time = 10:00:02.000 (NTP correction moves clock backwards by 500ms)
        System clock now shows: 10:00:01.500

        Problem: Next ID would have timestamp = 10:00:01.500
        → Older than previous ID (10:00:01.000)
        → Violates time-ordering
        ```

        **Solution: Refuse to generate IDs until time catches up**

        ```python
        class SnowflakeGenerator:
            def generate_id(self) -> int:
                """Generate ID with clock backwards detection"""
                with self.lock:
                    timestamp = self._current_timestamp()

                    # Clock moved backwards - CRITICAL ERROR
                    if timestamp < self.last_timestamp:
                        skew_ms = self.last_timestamp - timestamp

                        # Small skew (< 5ms): Wait for time to catch up
                        if skew_ms < 5:
                            logger.warning(f"Clock skew: {skew_ms}ms, waiting...")
                            time.sleep(skew_ms / 1000)
                            timestamp = self._current_timestamp()
                        else:
                            # Large skew: Refuse to generate
                            error_msg = f"Clock moved backwards by {skew_ms}ms. Refusing to generate ID."
                            logger.error(error_msg)
                            raise ClockBackwardsException(error_msg)

                    # ... rest of ID generation
        ```

        **Alternative: Use backup sequence**

        ```python
        class SnowflakeGenerator:
            """Enhanced generator with backup sequence for clock issues"""

            def __init__(self, machine_id: int):
                # ... existing initialization
                self.backup_sequence = 0
                self.MAX_BACKUP_SEQUENCE = 1000  # Allow 1000 IDs during clock issues

            def generate_id(self) -> int:
                with self.lock:
                    timestamp = self._current_timestamp()

                    # Clock moved backwards
                    if timestamp < self.last_timestamp:
                        # Use backup sequence with old timestamp
                        if self.backup_sequence < self.MAX_BACKUP_SEQUENCE:
                            logger.warning(f"Using backup sequence: {self.backup_sequence}")
                            id = self._generate_with_backup(self.last_timestamp)
                            self.backup_sequence += 1
                            return id
                        else:
                            raise ClockBackwardsException("Backup sequence exhausted")
                    else:
                        # Clock recovered
                        self.backup_sequence = 0

                    # ... normal ID generation

            def _generate_with_backup(self, timestamp: int) -> int:
                """Generate ID using backup sequence"""
                # Use higher bits of sequence for backup
                backup_id = (
                    ((timestamp - self.EPOCH) << self.TIMESTAMP_SHIFT) |
                    (self.machine_id << self.MACHINE_ID_SHIFT) |
                    (0x800 | self.backup_sequence)  # Set bit 11 for backup IDs
                )
                return backup_id
        ```

        ---

        ## Clock Drift Monitoring

        **Metrics to track:**

        | Metric | Target | Alert Threshold |
        |--------|--------|-----------------|
        | **NTP Sync Status** | Online | Offline |
        | **Clock Drift** | < 10ms | > 100ms |
        | **Clock Backwards Events** | 0/day | > 1/day |
        | **Backup Sequence Usage** | 0% | > 5% |

        **Monitoring dashboard:**

        ```python
        from prometheus_client import Counter, Histogram, Gauge

        # Metrics
        clock_backwards_count = Counter(
            'snowflake_clock_backwards_total',
            'Number of times clock moved backwards'
        )

        clock_drift_gauge = Gauge(
            'snowflake_clock_drift_ms',
            'Current clock drift in milliseconds'
        )

        backup_sequence_usage = Counter(
            'snowflake_backup_sequence_total',
            'Number of IDs generated using backup sequence'
        )

        # In generator
        def generate_id(self) -> int:
            timestamp = self._current_timestamp()

            if timestamp < self.last_timestamp:
                clock_backwards_count.inc()
                # ... handle clock backwards

            # ... generate ID
        ```

    === "🏗️ High Availability"

        ## The Challenge

        **Problem:** ID generator must be available 99.99% of the time. Any downtime blocks all writes.

        **Requirements:**

        - No single point of failure
        - Survive node failures
        - Fast failover (< 1 second)
        - No data loss on crash

        ---

        ## In-Process Generation (No External Service)

        **Architecture:**

        ```
        Traditional approach (SPOF):
        App Server 1 ──┐
        App Server 2 ──┼──> Centralized ID Service (BOTTLENECK)
        App Server 3 ──┘

        Snowflake approach (No SPOF):
        App Server 1 ──> Local ID Generator (machine_id=1)
        App Server 2 ──> Local ID Generator (machine_id=2)
        App Server 3 ──> Local ID Generator (machine_id=3)
        ```

        **Benefits:**

        - **No network calls** - Sub-millisecond latency
        - **No SPOF** - Each node independent
        - **Linear scalability** - Add nodes without coordination
        - **High availability** - Failure of one node doesn't affect others

        **Implementation as library:**

        ```python
        # Django/Flask app initialization
        from snowflake import SnowflakeGenerator

        # Get unique machine ID from config/environment
        MACHINE_ID = int(os.environ.get('MACHINE_ID', 0))

        # Initialize generator at application startup (singleton)
        id_generator = SnowflakeGenerator(machine_id=MACHINE_ID)

        # Use in application code
        def create_tweet(user_id: int, content: str):
            tweet_id = id_generator.generate_id()  # < 1ms, no network

            db.execute(
                "INSERT INTO tweets (id, user_id, content, created_at) VALUES (%s, %s, %s, NOW())",
                (tweet_id, user_id, content)
            )

            return tweet_id
        ```

        ---

        ## Machine ID Management

        **Challenge:** Ensure each node has unique machine ID (0-1023).

        **Approaches:**

        ### 1. Static Configuration

        ```yaml
        # config/production.yml
        app_servers:
          - host: app-server-1.prod
            machine_id: 1
          - host: app-server-2.prod
            machine_id: 2
          - host: app-server-3.prod
            machine_id: 3
        ```

        **Pros:** Simple, predictable
        **Cons:** Manual management, doesn't scale beyond 1024 nodes

        ---

        ### 2. ZooKeeper-based Registration

        ```python
        import kazoo.client

        class MachineIDRegistry:
            """Manage machine IDs using ZooKeeper"""

            ZK_PATH = '/snowflake/machines'

            def __init__(self, zk_hosts: str):
                self.zk = kazoo.client.KazooClient(hosts=zk_hosts)
                self.zk.start()
                self.machine_id = None

            def register(self) -> int:
                """
                Register this node and get unique machine ID

                Returns:
                    Assigned machine ID (0-1023)
                """
                # Get hostname
                hostname = socket.gethostname()
                node_path = f"{self.ZK_PATH}/{hostname}"

                # Check if already registered
                if self.zk.exists(node_path):
                    data, _ = self.zk.get(node_path)
                    self.machine_id = int(data.decode())
                    logger.info(f"Using existing machine ID: {self.machine_id}")
                    return self.machine_id

                # Find available machine ID
                for machine_id in range(1024):
                    id_path = f"{self.ZK_PATH}/id_{machine_id}"

                    # Try to create ephemeral node
                    try:
                        self.zk.create(
                            id_path,
                            value=hostname.encode(),
                            ephemeral=True,
                            makepath=True
                        )

                        # Success - save to persistent node
                        self.zk.create(
                            node_path,
                            value=str(machine_id).encode(),
                            makepath=True
                        )

                        self.machine_id = machine_id
                        logger.info(f"Registered new machine ID: {machine_id}")
                        return machine_id

                    except kazoo.exceptions.NodeExistsError:
                        # ID already taken, try next
                        continue

                raise Exception("No available machine IDs (all 1024 in use)")

            def health_check(self) -> bool:
                """Check if machine ID is still valid"""
                if not self.machine_id:
                    return False

                id_path = f"{self.ZK_PATH}/id_{self.machine_id}"
                return self.zk.exists(id_path) is not None
        ```

        **Pros:** Automatic assignment, handles node failures
        **Cons:** Requires ZooKeeper (additional dependency)

        ---

        ### 3. Hybrid: Datacenter + Worker ID

        ```python
        class SnowflakeGenerator:
            """
            Enhanced generator with datacenter + worker ID

            Machine ID (10 bits) split as:
            - 5 bits: Datacenter ID (0-31) - 32 datacenters
            - 5 bits: Worker ID (0-31) - 32 workers per datacenter

            Total capacity: 32 × 32 = 1024 unique IDs
            """

            DATACENTER_ID_BITS = 5
            WORKER_ID_BITS = 5

            MAX_DATACENTER_ID = (1 << DATACENTER_ID_BITS) - 1  # 31
            MAX_WORKER_ID = (1 << WORKER_ID_BITS) - 1          # 31

            def __init__(self, datacenter_id: int, worker_id: int):
                """
                Initialize generator with datacenter and worker IDs

                Args:
                    datacenter_id: Datacenter ID (0-31)
                    worker_id: Worker ID within datacenter (0-31)
                """
                if datacenter_id < 0 or datacenter_id > self.MAX_DATACENTER_ID:
                    raise ValueError(f"Datacenter ID must be 0-{self.MAX_DATACENTER_ID}")

                if worker_id < 0 or worker_id > self.MAX_WORKER_ID:
                    raise ValueError(f"Worker ID must be 0-{self.MAX_WORKER_ID}")

                # Combine into machine ID
                self.machine_id = (datacenter_id << self.WORKER_ID_BITS) | worker_id
                self.datacenter_id = datacenter_id
                self.worker_id = worker_id

                # ... rest of initialization

            def decode_id(self, id: int) -> dict:
                """Decode ID with datacenter and worker info"""
                base_info = super().decode_id(id)

                machine_id = base_info['machine_id']
                datacenter_id = (machine_id >> self.WORKER_ID_BITS) & self.MAX_DATACENTER_ID
                worker_id = machine_id & self.MAX_WORKER_ID

                base_info['datacenter_id'] = datacenter_id
                base_info['worker_id'] = worker_id

                return base_info
        ```

        **Benefits:**

        - Easier management: 32 DCs × 32 workers
        - Geographic distribution visible in ID
        - Better debugging (know which DC generated ID)

        ---

        ## Failure Modes and Recovery

        | Failure Mode | Impact | Recovery |
        |--------------|--------|----------|
        | **Single node crash** | No impact on other nodes | Restart node, re-register machine ID |
        | **ZooKeeper down** | New nodes can't register | Use cached machine ID, manual assignment |
        | **Clock sync failure** | Node stops generating IDs | Alert ops, restart with time sync |
        | **Machine ID collision** | ID duplicates (CRITICAL) | Monitoring detects, kill duplicate node |
        | **All 1024 IDs used** | Can't add more nodes | Split datacenter/worker IDs, upgrade to 11-bit machine ID |

        **Collision detection:**

        ```python
        class CollisionDetector:
            """Detect ID collisions in distributed system"""

            def __init__(self, redis_client):
                self.redis = redis_client

            def check_collision(self, id: int, machine_id: int) -> bool:
                """
                Check if ID was already generated by another machine

                Returns:
                    True if collision detected (CRITICAL BUG)
                """
                key = f"snowflake:id:{id}"

                # Try to set with NX (only if not exists)
                existing_machine = self.redis.get(key)

                if existing_machine and int(existing_machine) != machine_id:
                    # COLLISION DETECTED
                    logger.critical(f"ID collision: {id} generated by machines {existing_machine} and {machine_id}")
                    # Alert immediately
                    return True

                # Store ID with machine ID (expire after 1 hour)
                self.redis.setex(key, 3600, machine_id)
                return False
        ```

    === "🆔 Alternative Approaches"

        ## The Challenge

        **Problem:** Snowflake isn't the only solution for unique IDs. What are the trade-offs?

        ---

        ## Comparison Table

        | Approach | Uniqueness | Sortable | Performance | Size | Use Case |
        |----------|-----------|----------|-------------|------|----------|
        | **Snowflake** | ✅ Guaranteed | ✅ Yes (time) | ⚡ 1M/sec | 64-bit | **Distributed systems** |
        | **UUID v4** | ✅ Probabilistic | ❌ No | ⚡ 10M/sec | 128-bit | Offline generation |
        | **UUID v7** | ✅ Probabilistic | ✅ Yes (time) | ⚡ 10M/sec | 128-bit | Modern distributed systems |
        | **MongoDB ObjectID** | ✅ Guaranteed | ✅ Yes (time) | ⚡ 1M/sec | 96-bit | MongoDB databases |
        | **Database Auto-Increment** | ✅ Guaranteed | ✅ Yes | 🐌 10K/sec | 64-bit | Single database |
        | **ULID** | ✅ Probabilistic | ✅ Yes (time) | ⚡ 5M/sec | 128-bit | URL-safe, human-readable |

        ---

        ## 1. UUID (Universally Unique Identifier)

        ### UUID v4 (Random)

        ```python
        import uuid

        # Generate UUID v4 (random)
        id = uuid.uuid4()
        print(id)  # 550e8400-e29b-41d4-a716-446655440000

        # Properties:
        # - 128 bits (16 bytes)
        # - Probabilistic uniqueness (collision risk ~0 for practical purposes)
        # - NOT time-ordered
        # - No coordination required
        ```

        **Collision probability:**

        ```
        After generating 1 billion UUIDs:
        P(collision) = 1 - e^(-n²/2x) ≈ 0.0000000000000001
        (1 in 10^15 chance)
        ```

        **Pros:**

        - No coordination needed
        - Extremely fast generation
        - Standard across languages
        - Cryptographically secure

        **Cons:**

        - 128-bit (16 bytes) - 2x Snowflake size
        - NOT sortable - poor database indexing performance
        - Non-sequential - B-tree fragmentation
        - Not human-readable

        ---

        ### UUID v7 (Time-Ordered)

        ```python
        import uuid_utils as uuid

        # Generate UUID v7 (time-ordered, RFC draft)
        id = uuid.uuid7()
        print(id)  # 017F22E2-79B0-7CC3-98C4-DC0C0C07398F

        # Structure:
        # - 48 bits: Unix timestamp (milliseconds)
        # - 12 bits: Sub-millisecond precision
        # - 2 bits: Version/variant
        # - 62 bits: Random
        ```

        **Pros:**

        - Time-ordered (like Snowflake)
        - Standard UUID format
        - Better database indexing than UUID v4
        - No coordination needed

        **Cons:**

        - Still 128-bit (larger than Snowflake)
        - Probabilistic uniqueness (not guaranteed)
        - Not widely adopted yet (RFC draft)

        ---

        ## 2. MongoDB ObjectID

        ```python
        from bson import ObjectId

        # Generate ObjectID
        id = ObjectId()
        print(id)  # 507f1f77bcf86cd799439011

        # Structure (96 bits / 12 bytes):
        # - 32 bits: Unix timestamp (seconds)
        # - 24 bits: Machine identifier
        # - 16 bits: Process ID
        # - 24 bits: Counter
        ```

        **Implementation:**

        ```python
        import time
        import os
        import random
        import struct

        class ObjectIDGenerator:
            """MongoDB ObjectID generator"""

            def __init__(self):
                self.counter = random.randint(0, 0xFFFFFF)
                self.machine_id = self._generate_machine_id()
                self.process_id = os.getpid() & 0xFFFF

            def generate(self) -> bytes:
                """Generate 12-byte ObjectID"""
                # Timestamp (4 bytes)
                timestamp = int(time.time())

                # Increment counter (3 bytes)
                self.counter = (self.counter + 1) & 0xFFFFFF

                # Pack components
                object_id = struct.pack(
                    '>I',      # Timestamp (4 bytes, big-endian)
                    timestamp
                )
                object_id += self.machine_id  # Machine ID (3 bytes)
                object_id += struct.pack('>H', self.process_id)  # Process ID (2 bytes)
                object_id += struct.pack('>I', self.counter)[1:]  # Counter (3 bytes)

                return object_id

            def _generate_machine_id(self) -> bytes:
                """Generate 3-byte machine identifier from hostname"""
                import hashlib
                hostname = socket.gethostname().encode()
                hash_bytes = hashlib.md5(hostname).digest()
                return hash_bytes[:3]
        ```

        **Comparison with Snowflake:**

        | Feature | Snowflake | MongoDB ObjectID |
        |---------|-----------|------------------|
        | **Size** | 64-bit (8 bytes) | 96-bit (12 bytes) |
        | **Timestamp precision** | Millisecond | Second |
        | **Counter size** | 4096/ms | 16M/sec |
        | **Machine ID** | 1024 machines | 16M machines (hash-based) |
        | **Sortability** | Yes (ms precision) | Yes (second precision) |
        | **Uniqueness** | Guaranteed | Guaranteed |

        ---

        ## 3. Database Auto-Increment

        ```sql
        -- MySQL
        CREATE TABLE tweets (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            user_id BIGINT NOT NULL,
            content TEXT
        );

        -- Insert
        INSERT INTO tweets (user_id, content) VALUES (123, 'Hello');
        -- Returns id = 1

        -- PostgreSQL
        CREATE TABLE tweets (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            content TEXT
        );
        ```

        **Pros:**

        - Simple (database handles everything)
        - Guaranteed uniqueness
        - Sequential (optimal for indexing)
        - Small size (64-bit)

        **Cons:**

        - **Single point of failure** (database bottleneck)
        - **Poor horizontal scaling** (sharding breaks uniqueness)
        - **Reveals business metrics** (users see total count)
        - **Low throughput** (~10K/sec)
        - **Coordination overhead** (lock contention)

        **Sharding problem:**

        ```
        Shard 1: IDs 1, 4, 7, 10... (every 3rd)
        Shard 2: IDs 2, 5, 8, 11... (every 3rd)
        Shard 3: IDs 3, 6, 9, 12... (every 3rd)

        Problem: Cross-shard queries need to merge and sort
        ```

        ---

        ## 4. ULID (Universally Unique Lexicographically Sortable Identifier)

        ```python
        import ulid

        # Generate ULID
        id = ulid.new()
        print(id)  # 01ARZ3NDEKTSV4RRFFQ69G5FAV

        # Structure (128 bits / 16 bytes):
        # - 48 bits: Timestamp (milliseconds)
        # - 80 bits: Random

        # Properties:
        # - Lexicographically sortable (string comparison works)
        # - Case-insensitive (Crockford's Base32)
        # - URL-safe
        # - Human-readable
        ```

        **Implementation:**

        ```python
        import time
        import random

        class ULIDGenerator:
            """ULID generator"""

            # Crockford's Base32 alphabet (no I, L, O, U to avoid confusion)
            ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

            def generate(self) -> str:
                """Generate ULID string"""
                # Timestamp (48 bits)
                timestamp_ms = int(time.time() * 1000)

                # Random (80 bits)
                randomness = random.getrandbits(80)

                # Encode
                ulid_str = self._encode_timestamp(timestamp_ms) + self._encode_random(randomness)
                return ulid_str

            def _encode_timestamp(self, timestamp_ms: int) -> str:
                """Encode 48-bit timestamp to 10-char string"""
                result = ""
                for _ in range(10):
                    result = self.ENCODING[timestamp_ms & 0x1F] + result
                    timestamp_ms >>= 5
                return result

            def _encode_random(self, randomness: int) -> str:
                """Encode 80-bit random to 16-char string"""
                result = ""
                for _ in range(16):
                    result = self.ENCODING[randomness & 0x1F] + result
                    randomness >>= 5
                return result
        ```

        **Pros:**

        - Lexicographically sortable (string comparison)
        - URL-safe (no special characters)
        - Human-readable (26-char)
        - Millisecond precision
        - No coordination needed

        **Cons:**

        - 128-bit (larger than Snowflake)
        - Probabilistic uniqueness
        - String storage (less efficient than integer)

        ---

        ## 5. Instagram Sharded ID

        ```python
        class InstagramIDGenerator:
            """
            Instagram's ID scheme (pre-Snowflake)

            64-bit ID:
            - 41 bits: Timestamp (milliseconds)
            - 13 bits: Shard ID (8192 shards)
            - 10 bits: Auto-increment sequence per shard
            """

            EPOCH = 1314220021721  # Sept 1, 2011

            def __init__(self, shard_id: int):
                self.shard_id = shard_id
                self.sequences = {}  # timestamp -> sequence

            def generate_id(self, user_id: int) -> int:
                """Generate ID for user"""
                timestamp = int(time.time() * 1000) - self.EPOCH

                # Determine logical shard from user_id
                logical_shard = user_id % 8192

                # Get next sequence for this timestamp
                sequence = self.sequences.get(timestamp, 0)
                self.sequences[timestamp] = (sequence + 1) & 0x3FF

                # Construct ID
                id = (timestamp << 23) | (logical_shard << 10) | sequence
                return id
        ```

        **Benefits:**

        - Time-ordered
        - Logical shard ID embedded (routing hints)
        - Auto-increment per shard (sequential on shard)

        ---

        ## Recommendation Table

        | Use Case | Recommended | Reason |
        |----------|------------|--------|
        | **High-throughput distributed system** | Snowflake | Guaranteed uniqueness, time-ordered, 64-bit |
        | **Offline/client-side generation** | UUID v4 | No coordination, standard, secure |
        | **Modern distributed system (2024+)** | UUID v7 | Standard, time-ordered, 128-bit |
        | **MongoDB application** | ObjectID | Native support, time-ordered |
        | **Single database** | Auto-increment | Simple, sequential |
        | **URL sharing, human-readable** | ULID | Sortable, URL-safe, readable |
        | **Time-series data** | Snowflake/ULID | Time-ordering critical |

    === "⚠️ Edge Cases & Gotchas"

        ## The Challenge

        **Problem:** Real-world distributed systems have edge cases that break naive implementations.

        ---

        ## 1. Clock Synchronization Failure

        **Scenario:**

        ```
        Node A: Clock shows 10:00:00
        Node B: Clock shows 09:59:50 (10 seconds behind)

        User creates tweet on Node A: ID = 1000000000
        Same user creates another tweet on Node B: ID = 999500000

        Problem: Second tweet has SMALLER ID
        → Breaks time-ordering
        → Range queries miss records
        ```

        **Detection:**

        ```python
        def validate_id_ordering(user_id: int, new_id: int):
            """Validate ID is larger than user's last ID"""
            last_id = cache.get(f"user:{user_id}:last_id")

            if last_id and new_id <= last_id:
                logger.critical(f"ID ordering violation for user {user_id}: {new_id} <= {last_id}")
                # Alert immediately
                monitoring.alert("ID_ORDERING_VIOLATION", {
                    'user_id': user_id,
                    'new_id': new_id,
                    'last_id': last_id
                })

            cache.set(f"user:{user_id}:last_id", new_id, ttl=3600)
        ```

        ---

        ## 2. Sequence Overflow

        **Scenario:**

        ```
        Burst traffic: 5000 IDs requested in 1 millisecond
        Snowflake sequence: 0-4095 (4096 IDs per ms)

        Request 1-4096: OK (sequence 0-4095)
        Request 4097: Sequence overflow!

        Solution: Wait for next millisecond
        ```

        **Impact:**

        ```python
        # Without handling:
        # Throughput: 4096 IDs/ms = 4.096M IDs/sec
        # With waiting:
        # Average latency increases from 0.1ms to 0.5ms during bursts
        ```

        **Mitigation:**

        ```python
        class SnowflakeGenerator:
            def generate_id(self) -> int:
                with self.lock:
                    timestamp = self._current_timestamp()

                    if timestamp == self.last_timestamp:
                        self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE

                        if self.sequence == 0:
                            # Sequence overflow - wait
                            wait_start = time.time()
                            timestamp = self._wait_next_millis(self.last_timestamp)
                            wait_ms = (time.time() - wait_start) * 1000

                            # Track overflow events
                            sequence_overflow_count.inc()
                            sequence_overflow_wait_ms.observe(wait_ms)

                            if wait_ms > 5:
                                logger.warning(f"Sequence overflow wait: {wait_ms:.2f}ms")
                    # ... rest of generation
        ```

        ---

        ## 3. Machine ID Collision

        **Scenario:**

        ```
        Deployment mistake: Two nodes assigned machine_id = 5

        Node A generates: 1000000000005000
        Node B generates: 1000000000005001

        Same timestamp, same machine ID, different sequence
        → Duplicate IDs if sequences align
        ```

        **Detection:**

        ```python
        class CollisionDetector:
            """Detect duplicate machine IDs in production"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.heartbeat_interval = 10  # seconds

            def register_heartbeat(self, machine_id: int, hostname: str):
                """Register machine ID with hostname"""
                key = f"snowflake:machine:{machine_id}"

                # Check existing registration
                existing = self.redis.get(key)
                if existing:
                    existing_hostname = existing.decode()
                    if existing_hostname != hostname:
                        # COLLISION DETECTED
                        logger.critical(
                            f"Machine ID collision: {machine_id} used by both "
                            f"{existing_hostname} and {hostname}"
                        )
                        raise MachineIDCollisionException(
                            f"Machine ID {machine_id} already in use by {existing_hostname}"
                        )

                # Register with TTL
                self.redis.setex(
                    key,
                    self.heartbeat_interval * 3,  # 30 seconds
                    hostname.encode()
                )

            def start_heartbeat(self, machine_id: int, hostname: str):
                """Start heartbeat thread"""
                def heartbeat_loop():
                    while True:
                        try:
                            self.register_heartbeat(machine_id, hostname)
                            time.sleep(self.heartbeat_interval)
                        except Exception as e:
                            logger.error(f"Heartbeat failed: {e}")
                            time.sleep(1)

                thread = threading.Thread(target=heartbeat_loop, daemon=True)
                thread.start()
        ```

        ---

        ## 4. Timestamp Exhaustion

        **Scenario:**

        ```
        41-bit timestamp with custom epoch (Jan 1, 2020):
        Max value: 2^41 ms = 2,199,023,255,552 ms = 69.7 years

        IDs valid until: 2020 + 69 = 2089

        Problem: What happens in 2089?
        ```

        **Solution: Plan for migration**

        ```python
        class SnowflakeGenerator:
            """Generator with epoch monitoring"""

            EPOCH = 1577836800000  # Jan 1, 2020
            MAX_TIMESTAMP = (1 << 41) - 1  # 69 years

            def generate_id(self) -> int:
                timestamp = self._current_timestamp()
                elapsed = timestamp - self.EPOCH

                # Check if approaching exhaustion
                remaining_years = (self.MAX_TIMESTAMP - elapsed) / (365.25 * 24 * 3600 * 1000)

                if remaining_years < 5:
                    logger.warning(f"Snowflake timestamp exhaustion in {remaining_years:.1f} years")

                if elapsed > self.MAX_TIMESTAMP:
                    raise TimestampExhaustionException(
                        "Snowflake timestamp exhausted. Upgrade to new epoch required."
                    )

                # ... generate ID
        ```

        **Migration strategy:**

        - Use 42-bit timestamp (138 years)
        - Reduce machine ID to 9 bits (512 nodes)
        - OR: Migrate to UUID v7 (48-bit timestamp = 8000+ years)

        ---

        ## 5. Sharding and Rebalancing

        **Scenario:**

        ```
        Application shards by ID:
        Shard 1: IDs 0-1000000000
        Shard 2: IDs 1000000001-2000000000

        Problem: Snowflake IDs include timestamp
        → All new IDs go to latest shard
        → Unbalanced load
        ```

        **Solution: Shard by hash of ID**

        ```python
        def get_shard(id: int, num_shards: int) -> int:
            """Get shard for ID using consistent hash"""
            # Use last N bits (machine ID + sequence)
            # These bits are well-distributed
            shard_bits = id & 0x3FFFFF  # Last 22 bits
            return shard_bits % num_shards
        ```

        ---

        ## 6. Leap Seconds

        **Scenario:**

        ```
        June 30, 2026 23:59:59
        June 30, 2026 23:59:60  ← Leap second inserted
        July 1, 2026 00:00:00

        Problem: time.time() may repeat or skip
        ```

        **Mitigation:**

        - Use monotonic clock: `time.monotonic()` (doesn't jump)
        - Or: Smear leap seconds (Google's approach)
        - Or: Accept 1-second skew once per 1-2 years

        ---

        ## Summary: Production Checklist

        ✅ **Clock synchronization**
        - [ ] NTP configured on all nodes
        - [ ] Clock drift monitoring (< 100ms)
        - [ ] Alert on clock backwards events

        ✅ **Machine ID management**
        - [ ] Unique machine IDs guaranteed
        - [ ] ZooKeeper registration or static config
        - [ ] Collision detection in production

        ✅ **Monitoring**
        - [ ] IDs/sec per node
        - [ ] Sequence overflow rate
        - [ ] Clock drift metrics
        - [ ] Machine ID collision alerts

        ✅ **Failure handling**
        - [ ] Refuse IDs on clock backwards
        - [ ] Backup sequence for small clock skew
        - [ ] Graceful degradation on ZK failure

        ✅ **Testing**
        - [ ] Load test: 1M IDs/sec
        - [ ] Clock skew simulation
        - [ ] Machine ID collision test
        - [ ] Sequence overflow under load

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Snowflake from single node to 1000+ nodes generating billions of IDs per day.

    **Scaling challenges at 1000 nodes:**

    - **Throughput:** 1B IDs/sec total capacity
    - **Clock synchronization:** Sub-100ms drift across 1000 nodes
    - **Machine ID management:** 1024 unique IDs
    - **Monitoring:** Track health of 1000 generators

    ---

    ## Performance Optimization

    ### 1. Lock-Free Implementation

    **Problem:** Mutex lock becomes bottleneck at >1M IDs/sec.

    **Solution: Lock-free atomic operations**

    ```python
    import threading
    import ctypes

    class LockFreeSnowflakeGenerator:
        """Lock-free Snowflake generator using atomic operations"""

        def __init__(self, machine_id: int):
            self.machine_id = machine_id

            # State packed into single 64-bit integer:
            # - Upper 48 bits: Last timestamp
            # - Lower 16 bits: Sequence
            self.state = ctypes.c_uint64(0)

        def generate_id(self) -> int:
            """Generate ID using atomic compare-and-swap"""
            while True:
                # Read current state
                old_state = self.state.value
                old_timestamp = old_state >> 16
                old_sequence = old_state & 0xFFFF

                # Get current timestamp
                current_timestamp = self._current_timestamp()

                # Calculate new state
                if current_timestamp == old_timestamp:
                    # Same millisecond - increment sequence
                    new_sequence = (old_sequence + 1) & 0xFFF

                    if new_sequence == 0:
                        # Sequence overflow - wait
                        continue
                else:
                    # New millisecond - reset sequence
                    new_sequence = 0

                new_state = (current_timestamp << 16) | new_sequence

                # Atomic compare-and-swap
                if self._cas(old_state, new_state):
                    # Success - construct ID
                    timestamp_offset = current_timestamp - self.EPOCH
                    id = (
                        (timestamp_offset << self.TIMESTAMP_SHIFT) |
                        (self.machine_id << self.MACHINE_ID_SHIFT) |
                        new_sequence
                    )
                    return id
                # Failed - retry (another thread updated state)

        def _cas(self, old: int, new: int) -> bool:
            """Atomic compare-and-swap"""
            # In Python, use threading.Lock
            # In Go/Rust/C++, use atomic CAS instruction
            return ctypes.windll.kernel32.InterlockedCompareExchange64(
                ctypes.byref(self.state),
                new,
                old
            ) == old
    ```

    **Performance improvement:**

    - Locked: 800K IDs/sec (lock contention)
    - Lock-free: 2M IDs/sec (no contention)

    ---

    ### 2. Batch Generation

    **Problem:** Generating IDs one-at-a-time in loop is inefficient.

    **Solution: Pre-allocate ranges**

    ```python
    class BatchSnowflakeGenerator:
        """Generator that pre-allocates ID ranges"""

        BATCH_SIZE = 1000

        def __init__(self, machine_id: int):
            self.machine_id = machine_id
            self.id_buffer = []
            self.lock = threading.Lock()

        def generate_id(self) -> int:
            """Get ID from buffer or allocate new batch"""
            if not self.id_buffer:
                with self.lock:
                    # Double-check after acquiring lock
                    if not self.id_buffer:
                        self._allocate_batch()

            return self.id_buffer.pop()

        def _allocate_batch(self):
            """Pre-allocate batch of IDs"""
            timestamp = self._current_timestamp()

            # Generate batch
            for i in range(self.BATCH_SIZE):
                sequence = (self.sequence + i) & self.MAX_SEQUENCE

                id = (
                    ((timestamp - self.EPOCH) << self.TIMESTAMP_SHIFT) |
                    (self.machine_id << self.MACHINE_ID_SHIFT) |
                    sequence
                )
                self.id_buffer.append(id)

            self.sequence = (self.sequence + self.BATCH_SIZE) & self.MAX_SEQUENCE
    ```

    **Use case:**

    ```python
    # Bulk insert
    tweets = [
        {
            'id': generator.generate_id(),
            'user_id': user_id,
            'content': content
        }
        for content in tweet_contents
    ]

    db.bulk_insert('tweets', tweets)
    ```

    ---

    ### 3. Language-Specific Optimizations

    **Go (high-performance production implementation):**

    ```go
    package snowflake

    import (
        "sync"
        "time"
    )

    type Generator struct {
        mu            sync.Mutex
        epoch         int64
        machineID     int64
        sequence      int64
        lastTimestamp int64
    }

    func NewGenerator(machineID int64) *Generator {
        return &Generator{
            epoch:     1577836800000, // Jan 1, 2020
            machineID: machineID,
        }
    }

    func (g *Generator) Generate() int64 {
        g.mu.Lock()
        defer g.mu.Unlock()

        timestamp := time.Now().UnixNano() / 1e6

        if timestamp < g.lastTimestamp {
            panic("clock moved backwards")
        }

        if timestamp == g.lastTimestamp {
            g.sequence = (g.sequence + 1) & 0xFFF
            if g.sequence == 0 {
                // Wait for next millisecond
                timestamp = g.waitNextMillis(g.lastTimestamp)
            }
        } else {
            g.sequence = 0
        }

        g.lastTimestamp = timestamp

        id := ((timestamp - g.epoch) << 22) |
              (g.machineID << 12) |
              g.sequence

        return id
    }

    func (g *Generator) waitNextMillis(lastTimestamp int64) int64 {
        timestamp := time.Now().UnixNano() / 1e6
        for timestamp <= lastTimestamp {
            timestamp = time.Now().UnixNano() / 1e6
        }
        return timestamp
    }
    ```

    **Performance: 3M IDs/sec per node (Go vs 1M in Python)**

    ---

    ## Monitoring & Observability

    ### Key Metrics

    | Metric | Target | Alert Threshold | Action |
    |--------|--------|-----------------|--------|
    | **Generation Latency (P99)** | < 1ms | > 5ms | Check lock contention |
    | **IDs/sec per node** | 100K | < 10K | Check application load |
    | **Sequence overflow rate** | < 0.1% | > 5% | Increase nodes or reduce load |
    | **Clock drift** | < 10ms | > 100ms | Restart NTP sync |
    | **Clock backwards events** | 0/day | > 1/day | Check NTP config |
    | **Machine ID collisions** | 0 | > 0 | CRITICAL - kill duplicate |

    ---

    ### Prometheus Metrics

    ```python
    from prometheus_client import Counter, Histogram, Gauge

    # ID generation metrics
    ids_generated_total = Counter(
        'snowflake_ids_generated_total',
        'Total number of IDs generated',
        ['machine_id']
    )

    id_generation_duration_seconds = Histogram(
        'snowflake_id_generation_duration_seconds',
        'Time to generate single ID',
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01]  # 0.1ms to 10ms
    )

    sequence_overflow_total = Counter(
        'snowflake_sequence_overflow_total',
        'Number of sequence overflows (waited for next ms)',
        ['machine_id']
    )

    clock_drift_milliseconds = Gauge(
        'snowflake_clock_drift_milliseconds',
        'Current clock drift from NTP',
        ['machine_id']
    )

    clock_backwards_total = Counter(
        'snowflake_clock_backwards_total',
        'Number of times clock moved backwards',
        ['machine_id']
    )

    # Usage
    class SnowflakeGenerator:
        def generate_id(self) -> int:
            start = time.time()

            # ... generate ID

            ids_generated_total.labels(machine_id=self.machine_id).inc()
            id_generation_duration_seconds.observe(time.time() - start)

            return id
    ```

    ---

    ### Grafana Dashboard

    ```yaml
    # Example dashboard panels
    panels:
      - title: "IDs Generated/sec"
        query: rate(snowflake_ids_generated_total[1m])

      - title: "Generation Latency (P99)"
        query: histogram_quantile(0.99, snowflake_id_generation_duration_seconds)

      - title: "Sequence Overflow Rate"
        query: rate(snowflake_sequence_overflow_total[5m]) * 100

      - title: "Clock Drift"
        query: snowflake_clock_drift_milliseconds
        alert_threshold: 100

      - title: "Clock Backwards Events"
        query: increase(snowflake_clock_backwards_total[1h])
        alert_threshold: 1
    ```

    ---

    ## Cost Analysis

    ### Infrastructure Costs

    **For 1000-node deployment:**

    | Component | Cost/Month |
    |-----------|------------|
    | **Compute (1000 × t3.micro)** | $7,300 |
    | **ZooKeeper cluster (3 × m5.large)** | $216 |
    | **Redis (machine ID registry)** | $108 |
    | **NTP server (3 × t3.small)** | $54 |
    | **Monitoring (Prometheus + Grafana)** | $200 |
    | **Total** | **$7,878/month** |

    **Per-ID cost:**

    ```
    Monthly IDs: 110K/sec × 86,400 sec/day × 30 days = 285B IDs
    Cost per million IDs: $7,878 / 285,000 = $0.028

    Extremely cheap (< $0.03 per million IDs)
    ```

    ---

    ## Capacity Planning

    ### Current Load (Twitter-scale)

    ```
    Tweets: 6K IDs/sec
    Users: 500 IDs/sec
    Media: 2K IDs/sec
    Likes: 100K IDs/sec
    Total: 110K IDs/sec

    Required nodes (with headroom):
    - At 1M IDs/sec per node: 0.11 nodes
    - With 10x headroom: 2 nodes
    - Production deployment: 10 nodes (5 per datacenter)
    ```

    ### Growth Projection

    | Year | Daily Users | IDs/sec | Nodes Required |
    |------|-------------|---------|----------------|
    | 2026 | 400M | 110K | 10 |
    | 2028 | 600M | 165K | 15 |
    | 2030 | 1B | 275K | 25 |
    | 2035 | 2B | 550K | 50 |

    **Conclusion: Snowflake scales linearly with load. Add nodes as needed.**

    ---

    ## High Availability Configuration

    ### Multi-Datacenter Deployment

    ```
    Datacenter 1 (US-East):
    - Generators 1-340 (machine IDs 0-339)
    - ZooKeeper nodes 1-3
    - NTP server 1

    Datacenter 2 (US-West):
    - Generators 341-680 (machine IDs 340-679)
    - ZooKeeper nodes 4-6
    - NTP server 2

    Datacenter 3 (EU):
    - Generators 681-1000 (machine IDs 680-999)
    - ZooKeeper nodes 7-9
    - NTP server 3
    ```

    **Benefits:**

    - Survive datacenter failure (2/3 remain operational)
    - Low latency (generate IDs locally)
    - No cross-DC coordination needed

    ---

    ## Testing at Scale

    ### Load Test

    ```python
    import concurrent.futures
    import time

    def load_test(generator, duration_seconds=60, target_qps=1_000_000):
        """Load test: Generate IDs at target QPS"""
        start = time.time()
        ids_generated = 0
        errors = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            while time.time() - start < duration_seconds:
                # Submit batch of requests
                futures = [
                    executor.submit(generator.generate_id)
                    for _ in range(1000)
                ]

                # Wait for completion
                for future in concurrent.futures.as_completed(futures):
                    try:
                        id = future.result()
                        ids_generated += 1
                    except Exception as e:
                        errors += 1
                        print(f"Error: {e}")

                # Rate limiting (if needed)
                time.sleep(0.001)  # 1ms pause

        elapsed = time.time() - start
        actual_qps = ids_generated / elapsed

        print(f"Load Test Results:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  IDs generated: {ids_generated:,}")
        print(f"  Errors: {errors}")
        print(f"  Actual QPS: {actual_qps:,.0f}")
        print(f"  Target QPS: {target_qps:,}")
        print(f"  Success rate: {(1 - errors/ids_generated)*100:.2f}%")

    # Run test
    generator = SnowflakeGenerator(machine_id=123)
    load_test(generator, duration_seconds=60, target_qps=1_000_000)
    ```

    **Expected output:**

    ```
    Load Test Results:
      Duration: 60.00s
      IDs generated: 60,000,000
      Errors: 0
      Actual QPS: 1,000,000
      Target QPS: 1,000,000
      Success rate: 100.00%
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **64-bit structure:** Timestamp (41) + Machine ID (10) + Sequence (12)
    2. **No coordination:** Each node generates independently
    3. **Time-ordered:** IDs sortable by creation time (1ms resolution)
    4. **In-process generation:** Sub-millisecond latency, no network calls
    5. **ZooKeeper for machine IDs:** Prevent collisions, automatic registration
    6. **Clock backwards handling:** Refuse to generate, alert operators

    ---

    ## Interview Tips

    ✅ **Start with requirements** - Clarify uniqueness, sortability, performance

    ✅ **Explain bit layout** - Draw 64-bit structure, explain each component

    ✅ **Discuss alternatives** - Compare Snowflake, UUID, MongoDB ObjectID

    ✅ **Handle clock issues** - Clock drift, backwards movement, NTP sync

    ✅ **Scale to 1000 nodes** - Machine ID management, monitoring, HA

    ✅ **Mention trade-offs** - 64-bit vs 128-bit, time-ordered vs random

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"Why not use UUID?"** | UUIDs are 128-bit (2x size), not time-ordered, poor for database indexing |
    | **"How to handle clock moving backwards?"** | Refuse to generate IDs, wait for time to catch up, use backup sequence |
    | **"What if 1024 machine IDs isn't enough?"** | Split into datacenter (5 bits) + worker (5 bits), or use 11-bit machine ID |
    | **"How to ensure machine ID uniqueness?"** | ZooKeeper registration, ephemeral nodes, collision detection monitoring |
    | **"What if sequence overflows (>4095/ms)?"** | Wait for next millisecond (adds latency), or increase sequence bits |
    | **"Can IDs be decoded?"** | Yes, extract timestamp, machine ID, sequence using bit masks |
    | **"How to shard database by Snowflake ID?"** | Hash of ID (not range), use lower bits for good distribution |
    | **"What happens in 69 years?"** | Timestamp exhaustion, migrate to new epoch or larger timestamp |

    ---

    ## Code Template (Python)

    ```python
    import time
    import threading

    class SnowflakeGenerator:
        EPOCH = 1577836800000  # Jan 1, 2020 00:00:00 UTC
        TIMESTAMP_BITS = 41
        MACHINE_ID_BITS = 10
        SEQUENCE_BITS = 12

        MAX_MACHINE_ID = (1 << MACHINE_ID_BITS) - 1
        MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1

        TIMESTAMP_SHIFT = MACHINE_ID_BITS + SEQUENCE_BITS
        MACHINE_ID_SHIFT = SEQUENCE_BITS

        def __init__(self, machine_id: int):
            if not 0 <= machine_id <= self.MAX_MACHINE_ID:
                raise ValueError(f"Machine ID must be 0-{self.MAX_MACHINE_ID}")

            self.machine_id = machine_id
            self.sequence = 0
            self.last_timestamp = -1
            self.lock = threading.Lock()

        def generate_id(self) -> int:
            with self.lock:
                timestamp = int(time.time() * 1000)

                if timestamp < self.last_timestamp:
                    raise Exception("Clock moved backwards")

                if timestamp == self.last_timestamp:
                    self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE
                    if self.sequence == 0:
                        timestamp = self._wait_next_millis(self.last_timestamp)
                else:
                    self.sequence = 0

                self.last_timestamp = timestamp

                return (
                    ((timestamp - self.EPOCH) << self.TIMESTAMP_SHIFT) |
                    (self.machine_id << self.MACHINE_ID_SHIFT) |
                    self.sequence
                )

        def _wait_next_millis(self, last_timestamp: int) -> int:
            timestamp = int(time.time() * 1000)
            while timestamp <= last_timestamp:
                timestamp = int(time.time() * 1000)
            return timestamp
    ```

    ---

    ## Real-World Examples

    | Company | Implementation | Scale |
    |---------|---------------|-------|
    | **Twitter** | Original Snowflake | 400M users, billions of IDs/day |
    | **Instagram** | Modified Snowflake (13-bit shard ID) | 2B users |
    | **Discord** | Snowflake with epoch 2015 | Billions of messages |
    | **MongoDB** | ObjectID (similar concept, 96-bit) | Millions of deployments |
    | **Snowflake DB** | Named after algorithm, uses similar IDs | Petabyte-scale data warehouse |

    ---

    ## Additional Resources

    - Twitter Snowflake announcement: [https://blog.twitter.com/engineering/en_us/a/2010/announcing-snowflake](https://blog.twitter.com/engineering/en_us/a/2010/announcing-snowflake)
    - Instagram ID sharding: [https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c)
    - MongoDB ObjectID: [https://docs.mongodb.com/manual/reference/method/ObjectId/](https://docs.mongodb.com/manual/reference/method/ObjectId/)
    - UUID v7 draft: [https://datatracker.ietf.org/doc/html/draft-peabody-dispatch-new-uuid-format](https://datatracker.ietf.org/doc/html/draft-peabody-dispatch-new-uuid-format)

---

**Difficulty:** 🟢 Easy | **Interview Time:** 30-45 minutes | **Companies:** Twitter, Instagram, Discord, MongoDB, Snowflake

---

*Master this problem and you'll be ready for: Any distributed system requiring unique identifiers (social media, messaging, e-commerce, IoT)*
