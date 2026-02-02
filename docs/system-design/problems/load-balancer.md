# Design Load Balancer

A critical infrastructure component that distributes incoming network traffic across multiple backend servers to ensure high availability, optimal resource utilization, and minimal response time.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1M requests/sec, 1000 backend servers, 10k connections/sec |
| **Key Challenges** | Load distribution algorithms, health checks, session persistence, SSL termination, failover |
| **Core Concepts** | Round robin, least connections, consistent hashing, active/passive health checks, sticky sessions |
| **Companies** | HAProxy, NGINX, AWS ELB, F5 Networks, Cloudflare, Google Cloud, Azure, Netflix, Meta |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Traffic Distribution** | Distribute incoming requests across backend servers | P0 (Must have) |
    | **Health Monitoring** | Detect unhealthy servers and remove from pool | P0 (Must have) |
    | **Session Persistence** | Maintain user sessions on same server (sticky sessions) | P0 (Must have) |
    | **SSL Termination** | Handle SSL/TLS encryption/decryption | P0 (Must have) |
    | **Failover** | Automatically route traffic away from failed servers | P0 (Must have) |
    | **Connection Pooling** | Reuse backend connections for efficiency | P1 (Should have) |
    | **Rate Limiting** | Prevent abuse and DDoS attacks | P1 (Should have) |
    | **URL Routing** | Route based on URL patterns (path-based) | P1 (Should have) |
    | **WebSocket Support** | Handle persistent WebSocket connections | P2 (Nice to have) |
    | **Request Retries** | Retry failed requests on different servers | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Application-level logic (WAF, authentication)
    - Content caching (use CDN instead)
    - Advanced DDoS mitigation (use specialized services)
    - Service mesh features (use Istio/Linkerd)
    - Geographic load balancing (use DNS-based LB)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency Overhead** | < 1ms p95 | Load balancer should be transparent, minimal delay |
    | **Availability** | 99.99% uptime | Single point of failure, must be highly available |
    | **Throughput** | 1M requests/sec | Handle massive concurrent traffic |
    | **Connection Rate** | 10k new connections/sec | Support high connection churn |
    | **Scalability** | Linear scaling to 1000 servers | Add servers without performance degradation |
    | **Failover Time** | < 5 seconds | Quickly detect and route away from failures |
    | **CPU Efficiency** | < 20% CPU at max load | Minimize resource usage for cost efficiency |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Request volume:
    - Peak requests: 1M requests/sec
    - Average request size: 2 KB (headers + small payload)
    - Average response size: 50 KB (HTML/JSON)
    - Connection rate: 10k new connections/sec (100ms avg connection lifetime × 1M req/s = 10k conn/s)

    Backend servers:
    - Backend capacity: 1000 requests/sec per server (typical)
    - Servers needed: 1M / 1k = 1000 backend servers
    - Average connections per server: 100-500 concurrent

    Load balancer instances:
    - Throughput per LB: 100k requests/sec (typical hardware)
    - LB instances needed: 1M / 100k = 10 LB instances (for redundancy: 20 instances)
    - Active-active deployment for high availability
    ```

    ### Bandwidth Estimates

    ```
    Ingress (client -> load balancer):
    - Requests: 1M req/s × 2 KB = 2 GB/sec = 16 Gbps
    - WebSocket/persistent: 10% of connections = 100k × 1 KB/s = 100 MB/sec = 0.8 Gbps
    - Total ingress: ~17 Gbps

    Egress (load balancer -> client):
    - Responses: 1M req/s × 50 KB = 50 GB/sec = 400 Gbps
    - Total egress: ~400 Gbps

    Backend traffic (load balancer -> backend):
    - Ingress to backends: 17 Gbps (pass-through)
    - Egress from backends: 400 Gbps (pass-through)
    - Additional health check traffic: negligible (~1 Mbps)

    Note: For SSL termination, add ~20% CPU overhead
    ```

    ### Memory Estimates

    ```
    Connection tracking:
    - Active connections: 100k concurrent
    - Memory per connection: 64 KB (TCP buffers, state)
    - Total: 100k × 64 KB = 6.4 GB

    Session persistence (sticky sessions):
    - Active sessions: 1M (10% of daily users)
    - Memory per session: 256 bytes (session ID, server mapping)
    - Total: 1M × 256 B = 256 MB

    Health check state:
    - Servers: 1000
    - Health check data: 10 KB per server (status, latency history)
    - Total: 1000 × 10 KB = 10 MB

    SSL session cache:
    - SSL sessions: 100k concurrent
    - Memory per session: 4 KB (session keys, certificates)
    - Total: 100k × 4 KB = 400 MB

    Total memory per LB: 6.4 GB + 256 MB + 10 MB + 400 MB ≈ 7.1 GB
    ```

    ### CPU Estimates

    ```
    Operations per request:
    - Packet processing: 0.01ms
    - Load balancing decision: 0.05ms
    - SSL termination: 0.5ms (if enabled)
    - Health check processing: 0.02ms
    - Total per request: ~0.08ms (no SSL) or ~0.58ms (with SSL)

    CPU cores needed:
    - Without SSL: 1M req/s × 0.08ms = 80k ms/s = 80 cores
    - With SSL: 1M req/s × 0.58ms = 580k ms/s = 580 cores
    - With 20% overhead: ~96 cores (no SSL) or ~696 cores (with SSL)

    Per LB instance (100k req/s):
    - Without SSL: ~10 cores
    - With SSL: ~70 cores (or use hardware SSL acceleration)
    ```

    ---

    ## Key Assumptions

    1. Requests are mostly stateless (10% require sticky sessions)
    2. Backend servers are homogeneous (same capacity)
    3. Health checks run every 5 seconds
    4. Average connection lifetime is 100ms (short-lived HTTP)
    5. 30% of traffic is HTTPS requiring SSL termination
    6. Graceful degradation acceptable during failures

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **High availability:** Active-active deployment with automatic failover
    2. **Low latency:** Minimize overhead (< 1ms), use hardware acceleration
    3. **Scalability:** Horizontal scaling for both LB and backend servers
    4. **Transparency:** Clients unaware of backend changes
    5. **Health-first:** Proactive health monitoring, fast failure detection

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client1[Web Browser]
            Client2[Mobile App]
            Client3[API Client]
        end

        subgraph "DNS Layer"
            DNS[DNS Server<br/>Round-robin DNS<br/>Health-aware]
        end

        subgraph "Load Balancer Layer (Layer 4 + Layer 7)"
            LB1[Load Balancer 1<br/>Active<br/>100k req/s]
            LB2[Load Balancer 2<br/>Active<br/>100k req/s]
            LB3[Load Balancer N<br/>Standby<br/>Failover]

            subgraph "LB Components"
                SSL[SSL Terminator<br/>TLS 1.3]
                LB_Algo[Load Balancing<br/>Algorithm]
                HC[Health Checker<br/>Active/Passive]
                Session[Session Store<br/>Redis/Sticky]
                Metrics[Metrics<br/>Prometheus]
            end
        end

        subgraph "Backend Server Pool"
            BE1[Backend Server 1<br/>Healthy<br/>CPU: 40%]
            BE2[Backend Server 2<br/>Healthy<br/>CPU: 35%]
            BE3[Backend Server 3<br/>Degraded<br/>CPU: 85%]
            BE4[Backend Server N<br/>Unhealthy<br/>Removed]
        end

        subgraph "Health Check System"
            HC_Service[Health Check Service<br/>Active checks: HTTP<br/>Passive checks: Failure tracking]
            HC_DB[(Health State DB<br/>Redis<br/>Server status)]
        end

        subgraph "Session Persistence"
            Redis[Redis Cluster<br/>Session mapping<br/>Session ID -> Server]
        end

        subgraph "Configuration"
            Config[Config Store<br/>etcd/Consul<br/>Server list, algorithms]
        end

        subgraph "Monitoring"
            Monitor[Monitoring<br/>Grafana + Prometheus<br/>Latency, throughput, errors]
            Logs[Logs<br/>ELK Stack<br/>Request/error logs]
        end

        Client1 --> DNS
        Client2 --> DNS
        Client3 --> DNS

        DNS --> LB1
        DNS --> LB2
        DNS -.Failover.-> LB3

        LB1 --> SSL
        LB2 --> SSL

        SSL --> LB_Algo
        LB_Algo --> HC
        HC --> HC_Service
        LB_Algo --> Session
        Session --> Redis

        LB_Algo --> BE1
        LB_Algo --> BE2
        LB_Algo --> BE3
        LB_Algo -.X.-> BE4

        HC_Service --> HC_DB
        HC_Service --> BE1
        HC_Service --> BE2
        HC_Service --> BE3
        HC_Service --> BE4

        LB_Algo --> Metrics
        Metrics --> Monitor

        LB_Algo --> Config
        Config --> LB1
        Config --> LB2

        style LB1 fill:#e1f5ff
        style LB2 fill:#e1f5ff
        style LB3 fill:#fff4e1
        style BE1 fill:#e8f5e9
        style BE2 fill:#e8f5e9
        style BE3 fill:#fff9c4
        style BE4 fill:#ffebee
        style SSL fill:#e8eaf6
        style HC fill:#f3e5f5
        style Redis fill:#fff4e1
        style HC_DB fill:#fff4e1
        style Config fill:#e1f5e1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **DNS Round Robin** | First-level load balancing, distributes across LB instances | Anycast (complex routing), Single VIP (SPOF) |
    | **Layer 4 Load Balancer** | Low latency (<0.1ms), high throughput (1M req/s), TCP/UDP support | Layer 7 only (higher latency), Direct Server Return (complex) |
    | **Layer 7 Load Balancer** | Content-based routing, SSL termination, session persistence | Layer 4 only (no content awareness), Application-level proxy (too slow) |
    | **Redis for Sessions** | Fast session lookup (<1ms), distributed, persistent | In-memory only (lost on restart), Database (too slow), Sticky cookies (security risk) |
    | **Active Health Checks** | Proactive failure detection, prevents bad requests | Passive only (slower detection), No health checks (poor UX) |
    | **Consistent Hashing** | Minimal session disruption when servers added/removed | Random (session loss), Round robin (uneven load), IP hash (poor distribution) |

    **Key Trade-off:** We use **Layer 4 for speed + Layer 7 for features**. Pure Layer 4 is faster but lacks SSL termination and content routing. Hybrid approach balances performance and functionality.

    ---

    ## API Design

    ### 1. Health Check Endpoint (Backend)

    **Request:**
    ```http
    GET /health
    Host: backend-server-1.internal
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "status": "healthy",
      "timestamp": "2026-02-02T10:30:00Z",
      "checks": {
        "database": "ok",
        "cache": "ok",
        "disk": "ok"
      },
      "metrics": {
        "cpu_percent": 45,
        "memory_percent": 60,
        "active_connections": 120,
        "avg_response_time_ms": 35
      }
    }
    ```

    **Design Notes:**

    - Respond within 100ms (timeout threshold)
    - Include detailed health checks (DB, cache, disk)
    - Expose metrics for load balancing decisions
    - Return 503 if degraded (removed from pool)

    ---

    ### 2. Load Balancer Admin API

    **Add Server to Pool:**
    ```http
    POST /api/v1/servers
    Content-Type: application/json
    Authorization: Bearer <admin-token>

    {
      "server_id": "be-server-42",
      "ip": "10.0.1.42",
      "port": 8080,
      "weight": 100,
      "max_connections": 1000,
      "health_check_path": "/health"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "server_id": "be-server-42",
      "status": "draining",  // Gradual ramp-up
      "added_at": "2026-02-02T10:30:00Z",
      "health_status": "unknown",
      "message": "Server added to pool. Will receive traffic after 3 successful health checks."
    }
    ```

    ---

    ### 3. Get Load Balancer Status

    **Request:**
    ```http
    GET /api/v1/status
    Authorization: Bearer <admin-token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "load_balancer_id": "lb-001",
      "status": "active",
      "uptime_seconds": 86400,
      "metrics": {
        "requests_per_sec": 95000,
        "active_connections": 45000,
        "total_backend_servers": 1000,
        "healthy_servers": 987,
        "unhealthy_servers": 13,
        "avg_backend_latency_ms": 42,
        "cpu_percent": 18,
        "memory_percent": 35
      },
      "algorithm": "least_connections",
      "backends": [
        {
          "server_id": "be-server-1",
          "ip": "10.0.1.1",
          "status": "healthy",
          "active_connections": 45,
          "requests_per_sec": 950,
          "avg_latency_ms": 38,
          "health_check_consecutive_successes": 120
        },
        // ... more servers
      ]
    }
    ```

    ---

    ## Load Balancing Algorithms

    ### Algorithm Comparison

    | Algorithm | Description | Use Case | Pros | Cons |
    |-----------|-------------|----------|------|------|
    | **Round Robin** | Distribute requests sequentially | Equal server capacity, stateless | Simple, fair distribution | Ignores server load, sessions lost |
    | **Weighted Round Robin** | Round robin with server weights | Servers with different capacity | Respects capacity differences | Still ignores current load |
    | **Least Connections** | Route to server with fewest connections | Long-lived connections, varied request duration | Load-aware, balances better | Requires connection tracking, overhead |
    | **Least Response Time** | Route to server with lowest latency | Latency-sensitive apps | Best user experience | Complex tracking, noisy measurements |
    | **IP Hash** | Hash client IP to server | Stateful apps, session persistence | Session affinity, no shared state | Uneven distribution, session loss on server removal |
    | **Consistent Hashing** | Hash with virtual nodes | Session persistence, dynamic scaling | Minimal disruption, even distribution | Complex implementation, hash collisions |
    | **Random** | Random server selection | Stateless, simple apps | Very simple, low overhead | Poor distribution with few servers |

    ### Algorithm Selection Decision Tree

    ```mermaid
    graph TD
        Start[Select Algorithm] --> Stateful{Require Session<br/>Persistence?}

        Stateful -->|Yes| Sessions{Many Servers<br/>>100?}
        Sessions -->|Yes| ConsistentHash[Consistent Hashing<br/>✓ Minimal disruption<br/>✓ Even distribution]
        Sessions -->|No| IPHash[IP Hash<br/>✓ Simple<br/>⚠ Uneven load]

        Stateful -->|No| LongLived{Long-lived<br/>Connections?}

        LongLived -->|Yes| LeastConn[Least Connections<br/>✓ Load-aware<br/>✓ Better balance]

        LongLived -->|No| LatencySensitive{Latency<br/>Critical?}

        LatencySensitive -->|Yes| LeastLatency[Least Response Time<br/>✓ Best UX<br/>⚠ Complex]

        LatencySensitive -->|No| HeterogeneousServers{Servers Same<br/>Capacity?}

        HeterogeneousServers -->|Yes| RoundRobin[Round Robin<br/>✓ Simple<br/>✓ Fair]
        HeterogeneousServers -->|No| WeightedRR[Weighted Round Robin<br/>✓ Respects capacity<br/>✓ Simple]

        style ConsistentHash fill:#c8e6c9
        style LeastConn fill:#c8e6c9
        style RoundRobin fill:#c8e6c9
        style WeightedRR fill:#c8e6c9
        style IPHash fill:#fff9c4
        style LeastLatency fill:#fff9c4
    ```

    ---

    ## Database Schema

    ### Server Pool Configuration (etcd/Consul)

    ```json
    {
      "backend_servers": [
        {
          "server_id": "be-server-1",
          "ip": "10.0.1.1",
          "port": 8080,
          "weight": 100,
          "max_connections": 1000,
          "health_check": {
            "path": "/health",
            "interval_seconds": 5,
            "timeout_seconds": 2,
            "healthy_threshold": 3,
            "unhealthy_threshold": 2
          },
          "metadata": {
            "az": "us-east-1a",
            "instance_type": "c5.2xlarge",
            "created_at": "2026-01-15T10:00:00Z"
          }
        }
      ],
      "algorithm": "least_connections",
      "session_persistence": {
        "enabled": true,
        "type": "cookie",
        "cookie_name": "LB_SESSION",
        "ttl_seconds": 3600
      },
      "ssl": {
        "enabled": true,
        "certificate": "/etc/ssl/cert.pem",
        "private_key": "/etc/ssl/key.pem",
        "protocols": ["TLSv1.2", "TLSv1.3"]
      }
    }
    ```

    ---

    ### Health Check State (Redis)

    ```redis
    # Server health status
    HSET server:be-server-1 status healthy
    HSET server:be-server-1 last_check 1643712000
    HSET server:be-server-1 consecutive_successes 120
    HSET server:be-server-1 consecutive_failures 0
    HSET server:be-server-1 total_checks 1440
    HSET server:be-server-1 success_rate 99.3

    # Server metrics
    HSET server:be-server-1:metrics active_connections 45
    HSET server:be-server-1:metrics requests_per_sec 950
    HSET server:be-server-1:metrics avg_latency_ms 38
    HSET server:be-server-1:metrics cpu_percent 42
    HSET server:be-server-1:metrics memory_percent 60

    # Session persistence mapping
    HSET session:abc123def456 server_id be-server-1
    HSET session:abc123def456 created_at 1643712000
    EXPIRE session:abc123def456 3600

    # Time-series metrics (for trending)
    ZADD server:be-server-1:latency 1643712000 38
    ZADD server:be-server-1:latency 1643712005 42
    ZADD server:be-server-1:latency 1643712010 35
    ```

    ---

    ## Data Flow Diagrams

    ### Request Routing Flow (Layer 7)

    ```mermaid
    sequenceDiagram
        participant Client
        participant DNS
        participant LB as Load Balancer
        participant SSL as SSL Terminator
        participant Algo as LB Algorithm
        participant HC as Health Checker
        participant Session as Session Store
        participant BE1 as Backend Server 1
        participant BE2 as Backend Server 2

        Client->>DNS: Resolve api.example.com
        DNS-->>Client: LB IP: 203.0.113.10

        Client->>LB: HTTPS Request (GET /api/users)
        LB->>LB: Accept TCP connection

        LB->>SSL: Decrypt SSL/TLS
        SSL-->>LB: Plaintext HTTP

        LB->>LB: Parse HTTP headers (Host, Cookie, Path)

        alt Has session cookie
            LB->>Session: GET session:cookie_value
            Session-->>LB: server_id: be-server-1
            LB->>HC: Check server health
            HC-->>LB: be-server-1 is healthy
            LB->>BE1: Forward request
            BE1-->>LB: HTTP 200 OK (response)
        else No session cookie (new session)
            LB->>HC: Get healthy servers
            HC-->>LB: [be-server-1, be-server-2, ...]

            LB->>Algo: Select server (least connections)
            Algo->>Algo: Count connections per server
            Algo-->>LB: be-server-2 (35 connections)

            LB->>BE2: Forward request
            BE2-->>LB: HTTP 200 OK (response)

            LB->>Session: SET session:new_cookie be-server-2 EX 3600
        end

        LB->>SSL: Encrypt response
        SSL-->>Client: HTTPS Response + Set-Cookie: LB_SESSION=new_cookie
    ```

    **Flow Explanation:**

    1. **DNS resolution** - Client resolves domain to LB IP
    2. **TCP handshake** - Three-way handshake (SYN, SYN-ACK, ACK)
    3. **SSL termination** - Decrypt TLS, inspect HTTP headers
    4. **Session check** - Look up session cookie in Redis
    5. **Health verification** - Ensure target server is healthy
    6. **Algorithm selection** - Choose backend using configured algorithm
    7. **Request forwarding** - Proxy request to backend server
    8. **Response handling** - Encrypt and send back to client

    ---

    ### Health Check Flow

    ```mermaid
    sequenceDiagram
        participant Scheduler as Health Check Scheduler
        participant HC as Health Checker
        participant BE as Backend Server
        participant Redis as Health State Store
        participant LB as Load Balancer

        loop Every 5 seconds
            Scheduler->>HC: Trigger health check for all servers

            par Concurrent health checks
                HC->>BE: GET /health (timeout: 2s)
                BE-->>HC: HTTP 200 OK + health metrics
            end

            alt Health check success
                HC->>HC: Parse response, validate status
                HC->>Redis: HINCRBY server:be-1 consecutive_successes 1
                HC->>Redis: HSET server:be-1 consecutive_failures 0
                HC->>Redis: HSET server:be-1 last_check <timestamp>

                Redis->>Redis: Check consecutive_successes >= 3
                Redis-->>HC: Mark as healthy

                HC->>LB: Update server status: healthy
            else Health check failure (timeout or 5xx)
                HC->>Redis: HINCRBY server:be-1 consecutive_failures 1
                HC->>Redis: HSET server:be-1 consecutive_successes 0

                Redis->>Redis: Check consecutive_failures >= 2
                Redis-->>HC: Mark as unhealthy

                HC->>LB: Remove server from pool
                HC->>LB: Log alert: Server be-1 unhealthy
            end
        end

        Note over LB: Passive health check (real requests)
        LB->>BE: Forward client request
        BE-->>LB: Connection timeout / 502 Bad Gateway
        LB->>Redis: HINCRBY server:be-1 consecutive_failures 1
        LB->>LB: Retry request on different server
    ```

    **Health Check Types:**

    1. **Active Health Checks:**
        - Periodic HTTP/TCP checks (every 5s)
        - Configurable thresholds (3 success, 2 failures)
        - Proactive failure detection

    2. **Passive Health Checks:**
        - Monitor real request failures
        - Faster failure detection (no 5s wait)
        - Complement active checks

    ---

    ### Failover Flow (Server Failure)

    ```mermaid
    sequenceDiagram
        participant Client
        participant LB
        participant BE1 as Backend 1 (Healthy)
        participant BE2 as Backend 2 (Failing)
        participant BE3 as Backend 3 (Healthy)
        participant HC as Health Checker
        participant Redis

        Client->>LB: Request #1
        LB->>BE2: Forward to BE2
        BE2-->>LB: HTTP 200 OK
        LB-->>Client: Response

        Note over BE2: Server starts failing

        Client->>LB: Request #2
        LB->>BE2: Forward to BE2
        BE2--XLB: Connection timeout (2s)

        LB->>Redis: HINCRBY server:be-2 consecutive_failures 1
        LB->>BE1: Retry on BE1
        BE1-->>LB: HTTP 200 OK
        LB-->>Client: Response (3s total latency)

        Client->>LB: Request #3
        LB->>BE2: Forward to BE2
        BE2--XLB: Connection timeout

        LB->>Redis: HINCRBY server:be-2 consecutive_failures 2
        Redis->>Redis: consecutive_failures >= 2
        Redis-->>LB: Mark BE2 as unhealthy

        LB->>LB: Remove BE2 from active pool
        LB->>BE3: Retry on BE3
        BE3-->>LB: HTTP 200 OK
        LB-->>Client: Response

        Note over LB: BE2 removed, no more traffic

        loop Every 5 seconds
            HC->>BE2: Active health check
            BE2--XHC: Timeout / 503
            HC->>Redis: Keep BE2 as unhealthy
        end

        Note over BE2: Server recovers

        HC->>BE2: Health check
        BE2-->>HC: HTTP 200 OK
        HC->>Redis: HINCRBY server:be-2 consecutive_successes 1

        HC->>BE2: Health check #2
        BE2-->>HC: HTTP 200 OK
        HC->>Redis: HINCRBY server:be-2 consecutive_successes 2

        HC->>BE2: Health check #3
        BE2-->>HC: HTTP 200 OK
        HC->>Redis: HINCRBY server:be-2 consecutive_successes 3
        HC->>LB: Add BE2 back to pool (gradual ramp-up)

        LB->>LB: BE2 status: draining (10% traffic)
        Note over LB: Gradually increase to 100% over 1 minute
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical load balancer subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Layer 4 vs Layer 7** | When to use each? What are trade-offs? | Hybrid: L4 for speed, L7 for features |
    | **Consistent Hashing** | How to maintain sessions with dynamic servers? | Virtual nodes, minimal session disruption |
    | **Health Checks** | How to detect failures quickly and reliably? | Active + passive checks, exponential backoff |
    | **SSL Termination** | How to handle SSL at scale? | Hardware acceleration, session resumption |

    ---

    === "🔀 Layer 4 vs Layer 7"

        ## The Challenge

        **Problem:** Choose between Layer 4 (transport) and Layer 7 (application) load balancing. Each has trade-offs.

        **Layer 4 (Transport Layer):**
        - Operates on: TCP/UDP packets
        - Routing decisions: IP address, port
        - Performance: Very fast (<0.1ms latency)
        - Features: Limited (no content inspection)

        **Layer 7 (Application Layer):**
        - Operates on: HTTP/HTTPS requests
        - Routing decisions: URL, headers, cookies
        - Performance: Slower (~1ms latency)
        - Features: Rich (content-based routing, SSL termination)

        ---

        ## Layer 4 Load Balancing

        **How it works:**

        ```
        Client:12345 --> LB:80 --> Backend:8080

        LB forwards TCP packets:
        - Source IP: Client IP
        - Dest IP: Backend IP
        - No packet inspection
        - Fast forwarding (hardware accelerated)
        ```

        **Implementation (Direct Server Return - DSR):**

        ```python
        import socket
        import struct

        class Layer4LoadBalancer:
            """
            Layer 4 load balancer using Direct Server Return

            Client -> LB -> Backend (request)
            Client <- Backend (response, bypasses LB)
            """

            def __init__(self, backend_servers):
                self.backends = backend_servers
                self.current_index = 0

            def handle_connection(self, client_socket):
                """
                Accept connection and forward to backend

                Args:
                    client_socket: Client TCP socket
                """
                # Select backend (round robin)
                backend = self.backends[self.current_index % len(self.backends)]
                self.current_index += 1

                # Create connection to backend
                backend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                backend_socket.connect((backend['ip'], backend['port']))

                # Get original client IP and port
                client_ip, client_port = client_socket.getpeername()

                # Rewrite packet headers (DSR mode)
                # Backend sees original client IP, responds directly to client
                self._set_tproxy(backend_socket, client_ip, client_port)

                # Bidirectional forwarding
                self._forward_packets(client_socket, backend_socket)

            def _set_tproxy(self, backend_socket, client_ip, client_port):
                """
                Set TPROXY (Transparent Proxy) mode
                Backend server sees original client IP
                """
                # Set socket options for transparent proxy
                backend_socket.setsockopt(
                    socket.SOL_IP,
                    socket.IP_TRANSPARENT,
                    1
                )

                # Spoof source IP to client IP (requires root/CAP_NET_RAW)
                # This allows backend to respond directly to client
                pass  # Implementation omitted (requires iptables rules)

            def _forward_packets(self, client_sock, backend_sock):
                """Forward packets between client and backend"""
                import select

                sockets = [client_sock, backend_sock]

                while True:
                    readable, _, _ = select.select(sockets, [], [])

                    for sock in readable:
                        data = sock.recv(4096)
                        if not data:
                            return  # Connection closed

                        # Forward to opposite socket
                        if sock is client_sock:
                            backend_sock.sendall(data)
                        else:
                            client_sock.sendall(data)
        ```

        **Pros:**
        - **Ultra-low latency:** < 0.1ms overhead
        - **High throughput:** 1M+ req/s per instance
        - **Efficient:** Response bypasses LB (DSR mode)
        - **Simple:** No packet parsing, just forwarding

        **Cons:**
        - **No content awareness:** Can't route by URL, headers
        - **No SSL termination:** Backends must handle SSL
        - **Limited features:** No caching, compression, rate limiting
        - **Session persistence:** Only IP-based (IP hash)

        ---

        ## Layer 7 Load Balancing

        **How it works:**

        ```
        Client --> LB (parse HTTP) --> Backend

        LB inspects HTTP:
        - Headers: Host, User-Agent, Cookie
        - Path: /api/v1/users
        - Method: GET, POST, PUT
        - Body: JSON payload
        ```

        **Implementation (NGINX-style):**

        ```python
        import asyncio
        import aiohttp
        from urllib.parse import urlparse

        class Layer7LoadBalancer:
            """
            Layer 7 (HTTP) load balancer with content-based routing
            """

            def __init__(self, backend_servers, session_store):
                self.backends = backend_servers
                self.sessions = session_store
                self.connection_counts = {b['id']: 0 for b in backend_servers}

            async def handle_http_request(self, request):
                """
                Parse HTTP request and route to backend

                Args:
                    request: HTTP request object

                Returns:
                    HTTP response
                """
                # Parse request
                path = request.path
                method = request.method
                headers = request.headers
                cookies = request.cookies

                # Check session persistence (sticky sessions)
                session_id = cookies.get('LB_SESSION')
                if session_id:
                    backend = await self._get_session_backend(session_id)
                    if backend and self._is_healthy(backend):
                        return await self._forward_request(request, backend)

                # Content-based routing
                backend = self._route_by_content(path, method, headers)

                if not backend:
                    # Fallback to load balancing algorithm
                    backend = self._select_backend_least_connections()

                # Forward request
                response = await self._forward_request(request, backend)

                # Set session cookie if new session
                if not session_id:
                    session_id = self._generate_session_id()
                    await self.sessions.set(session_id, backend['id'])
                    response.set_cookie('LB_SESSION', session_id, max_age=3600)

                return response

            def _route_by_content(self, path, method, headers):
                """
                Route based on request content

                Examples:
                - /api/* -> API backend pool
                - /static/* -> Static file servers
                - Host: admin.example.com -> Admin backend
                """
                host = headers.get('Host', '')

                # Admin subdomain -> admin servers
                if 'admin.' in host:
                    return self._get_backend_by_tag('admin')

                # API requests -> API servers
                if path.startswith('/api/'):
                    return self._get_backend_by_tag('api')

                # Static files -> static servers
                if path.startswith('/static/'):
                    return self._get_backend_by_tag('static')

                # WebSocket upgrade -> WebSocket servers
                if headers.get('Upgrade', '').lower() == 'websocket':
                    return self._get_backend_by_tag('websocket')

                return None  # No content-based route

            def _select_backend_least_connections(self):
                """
                Select backend with least active connections

                Returns:
                    Backend server dict
                """
                healthy_backends = [b for b in self.backends if self._is_healthy(b)]

                if not healthy_backends:
                    raise Exception("No healthy backends available")

                # Find backend with minimum connections
                return min(
                    healthy_backends,
                    key=lambda b: self.connection_counts[b['id']]
                )

            async def _forward_request(self, request, backend):
                """
                Forward HTTP request to backend server

                Args:
                    request: Client HTTP request
                    backend: Target backend server

                Returns:
                    HTTP response from backend
                """
                self.connection_counts[backend['id']] += 1

                try:
                    async with aiohttp.ClientSession() as session:
                        # Forward request to backend
                        url = f"http://{backend['ip']}:{backend['port']}{request.path}"

                        async with session.request(
                            method=request.method,
                            url=url,
                            headers=self._prepare_headers(request.headers),
                            data=await request.read()
                        ) as backend_response:

                            # Read response
                            body = await backend_response.read()

                            return self._build_response(
                                status=backend_response.status,
                                headers=backend_response.headers,
                                body=body
                            )

                finally:
                    self.connection_counts[backend['id']] -= 1

            def _prepare_headers(self, client_headers):
                """
                Prepare headers for backend request
                Add X-Forwarded-* headers
                """
                headers = dict(client_headers)

                # Add load balancer headers
                headers['X-Forwarded-For'] = client_headers.get('X-Real-IP', 'unknown')
                headers['X-Forwarded-Proto'] = 'https'
                headers['X-Forwarded-Host'] = client_headers.get('Host', '')

                # Remove hop-by-hop headers
                for h in ['Connection', 'Keep-Alive', 'Proxy-Authenticate']:
                    headers.pop(h, None)

                return headers
        ```

        **Pros:**
        - **Content-based routing:** Route by URL, headers, cookies
        - **SSL termination:** Offload SSL from backends
        - **Advanced features:** Rate limiting, caching, compression
        - **Session persistence:** Cookie-based, more reliable

        **Cons:**
        - **Higher latency:** ~1ms overhead (HTTP parsing)
        - **Lower throughput:** ~100k req/s per instance
        - **More CPU:** HTTP parsing, SSL encryption
        - **Stateful:** Must track sessions, connections

        ---

        ## Hybrid Approach (Recommended)

        **Best of both worlds:**

        ```
        Client --> Layer 4 LB --> Layer 7 LB --> Backend
                   (fast routing)    (content routing)

        Layer 4: Health-based IP routing
        Layer 7: Content-based application routing
        ```

        **When to use each:**

        | Scenario | Layer 4 | Layer 7 | Hybrid |
        |----------|---------|---------|--------|
        | **TCP/UDP services** (databases, cache) | ✅ Best | ❌ N/A | ✅ L4 only |
        | **Simple HTTP** (stateless, no SSL) | ✅ Good | ✅ Good | ✅ L4 for speed |
        | **Content routing** (path-based) | ❌ Can't | ✅ Best | ✅ L7 required |
        | **SSL termination** | ❌ Can't | ✅ Best | ✅ L7 required |
        | **Session persistence** | 🟡 IP-only | ✅ Cookie | ✅ L7 for cookies |
        | **Ultra-low latency** (<1ms) | ✅ Best | ❌ ~1ms | 🟡 L4 primary |
        | **Very high throughput** (>500k req/s) | ✅ Best | ❌ ~100k | 🟡 L4 primary |

    === "🔑 Consistent Hashing"

        ## The Challenge

        **Problem:** Maintain session persistence when backend servers are added/removed. Naive hashing breaks sessions.

        **Naive approach (IP hash):**

        ```python
        def select_backend(client_ip, backends):
            index = hash(client_ip) % len(backends)
            return backends[index]
        ```

        **Problem with naive hash:**

        ```
        Initially: 3 servers
        hash(client_ip) % 3 = 1  -> Server 1

        After adding 1 server (4 total):
        hash(client_ip) % 4 = 2  -> Server 2 (DIFFERENT!)

        Result: Session lost, user logged out
        ```

        **Impact:** Adding/removing 1 server causes ~80% of sessions to remap!

        ---

        ## Consistent Hashing Solution

        **Key idea:** Map both servers and clients onto a hash ring. Client routed to next server clockwise.

        **Hash Ring:**

        ```
                        0
                        |
            270 --------|-------- 90
                        |
                       180

        Servers:
        - Server A: hash(A) = 45
        - Server B: hash(B) = 135
        - Server C: hash(C) = 225

        Clients:
        - Client 1: hash(IP1) = 30  -> next server clockwise = A (45)
        - Client 2: hash(IP2) = 100 -> next server clockwise = B (135)
        - Client 3: hash(IP3) = 200 -> next server clockwise = C (225)
        ```

        **When Server B removed:**

        ```
        - Client 1: 30 -> A (unchanged)
        - Client 2: 100 -> C (225) - remapped to nearest
        - Client 3: 200 -> C (unchanged)

        Result: Only 1 out of 3 clients affected (33%)
        ```

        ---

        ## Implementation with Virtual Nodes

        **Problem:** With few servers, hash distribution is uneven.

        **Solution:** Create virtual nodes (replicas) for each server.

        ```python
        import hashlib
        import bisect
        from typing import List, Dict

        class ConsistentHash:
            """
            Consistent hashing with virtual nodes for even distribution
            """

            def __init__(self, nodes: List[str], virtual_node_count: int = 150):
                """
                Args:
                    nodes: List of backend server IDs
                    virtual_node_count: Number of virtual nodes per physical server
                """
                self.virtual_node_count = virtual_node_count
                self.ring = {}  # hash -> server_id
                self.sorted_keys = []  # Sorted hash values
                self.node_stats = {}  # server_id -> request count

                for node in nodes:
                    self.add_node(node)

            def _hash(self, key: str) -> int:
                """
                Hash function (MD5 for good distribution)

                Returns:
                    Integer hash value (32-bit)
                """
                return int(hashlib.md5(key.encode()).hexdigest(), 16)

            def add_node(self, node_id: str):
                """
                Add server to hash ring with virtual nodes

                Args:
                    node_id: Backend server identifier
                """
                self.node_stats[node_id] = 0

                # Create virtual nodes
                for i in range(self.virtual_node_count):
                    virtual_key = f"{node_id}:vnode{i}"
                    hash_value = self._hash(virtual_key)

                    self.ring[hash_value] = node_id
                    bisect.insort(self.sorted_keys, hash_value)

                print(f"Added node {node_id} with {self.virtual_node_count} virtual nodes")

            def remove_node(self, node_id: str):
                """
                Remove server from hash ring

                Args:
                    node_id: Backend server identifier
                """
                # Remove all virtual nodes
                for i in range(self.virtual_node_count):
                    virtual_key = f"{node_id}:vnode{i}"
                    hash_value = self._hash(virtual_key)

                    del self.ring[hash_value]
                    self.sorted_keys.remove(hash_value)

                del self.node_stats[node_id]
                print(f"Removed node {node_id}")

            def get_node(self, key: str) -> str:
                """
                Get backend server for client key

                Args:
                    key: Client identifier (IP, session ID)

                Returns:
                    Backend server ID
                """
                if not self.ring:
                    raise Exception("No nodes in hash ring")

                # Hash client key
                hash_value = self._hash(key)

                # Find next server on ring (binary search)
                index = bisect.bisect_right(self.sorted_keys, hash_value)

                # Wrap around if at end of ring
                if index == len(self.sorted_keys):
                    index = 0

                server_hash = self.sorted_keys[index]
                node_id = self.ring[server_hash]

                # Track stats
                self.node_stats[node_id] += 1

                return node_id

            def get_stats(self) -> Dict[str, float]:
                """
                Get distribution statistics

                Returns:
                    Dict of server_id -> percentage of requests
                """
                total = sum(self.node_stats.values())
                if total == 0:
                    return {}

                return {
                    node: (count / total) * 100
                    for node, count in self.node_stats.items()
                }


        # Example usage
        if __name__ == "__main__":
            # Initialize with 3 servers
            servers = ["server-1", "server-2", "server-3"]
            ch = ConsistentHash(servers, virtual_node_count=150)

            # Simulate 10,000 client requests
            clients = [f"client-{i}" for i in range(10000)]

            # Track initial assignments
            initial_assignments = {}
            for client in clients:
                server = ch.get_node(client)
                initial_assignments[client] = server

            print("\n=== Initial Distribution ===")
            print(ch.get_stats())
            # Output: server-1: 33.2%, server-2: 33.5%, server-3: 33.3%

            # Add new server
            ch.add_node("server-4")

            # Re-check assignments
            remapped_count = 0
            for client in clients:
                new_server = ch.get_node(client)
                if new_server != initial_assignments[client]:
                    remapped_count += 1

            print(f"\n=== After Adding Server-4 ===")
            print(f"Remapped clients: {remapped_count} / {len(clients)} ({remapped_count/len(clients)*100:.1f}%)")
            print(ch.get_stats())
            # Output: ~25% remapped (ideal), distribution: ~25% each

            # Remove server
            ch.remove_node("server-2")

            print(f"\n=== After Removing Server-2 ===")
            print(ch.get_stats())
            # Output: server-1: 33%, server-3: 33%, server-4: 33%
        ```

        ---

        ## Consistent Hashing Trade-offs

        | Aspect | Consistent Hashing | Naive Hash | Random |
        |--------|-------------------|------------|--------|
        | **Session disruption on add/remove** | ✅ Minimal (~25%) | ❌ High (~80%) | ❌ Total (100%) |
        | **Load distribution** | ✅ Even (with vnodes) | ✅ Even | 🟡 Uneven with few servers |
        | **Lookup time** | 🟡 O(log N) binary search | ✅ O(1) | ✅ O(1) |
        | **Memory overhead** | 🟡 High (vnodes storage) | ✅ Low | ✅ Low |
        | **Implementation complexity** | 🟡 Complex (ring, vnodes) | ✅ Simple | ✅ Very simple |

        **Recommendation:**

        - **Use consistent hashing** for session-heavy apps with dynamic scaling
        - **Use IP hash** for simple stateful apps with static servers
        - **Use least connections** for stateless apps

    === "💓 Health Checks"

        ## The Challenge

        **Problem:** Detect backend server failures quickly and reliably without overloading servers with health checks.

        **Requirements:**

        - **Fast detection:** Detect failures within 5 seconds
        - **Low false positives:** Avoid removing healthy servers
        - **Scalable:** 1000 servers × 10 LB instances = 10k checks/sec
        - **Minimal overhead:** Health checks shouldn't impact server performance

        ---

        ## Active vs Passive Health Checks

        ### Active Health Checks

        **Load balancer proactively queries backend servers.**

        ```python
        import asyncio
        import aiohttp
        import time
        from typing import Dict, List
        from enum import Enum

        class HealthStatus(Enum):
            HEALTHY = "healthy"
            UNHEALTHY = "unhealthy"
            DRAINING = "draining"
            UNKNOWN = "unknown"

        class ActiveHealthChecker:
            """
            Active health checker with exponential backoff
            """

            def __init__(self, servers: List[Dict], redis_client):
                self.servers = servers
                self.redis = redis_client

                # Configuration
                self.check_interval = 5  # seconds
                self.timeout = 2  # seconds
                self.healthy_threshold = 3  # consecutive successes
                self.unhealthy_threshold = 2  # consecutive failures

                # State
                self.server_states = {}

            async def start_monitoring(self):
                """Start health check loop for all servers"""
                tasks = [
                    self.monitor_server(server)
                    for server in self.servers
                ]
                await asyncio.gather(*tasks)

            async def monitor_server(self, server: Dict):
                """
                Monitor single server with exponential backoff

                Args:
                    server: Server configuration dict
                """
                server_id = server['id']
                check_interval = self.check_interval

                while True:
                    try:
                        # Perform health check
                        is_healthy = await self.check_health(server)

                        # Update state
                        await self.update_health_state(server_id, is_healthy)

                        # Reset interval on success
                        if is_healthy:
                            check_interval = self.check_interval
                        else:
                            # Exponential backoff for unhealthy servers
                            check_interval = min(check_interval * 2, 60)

                    except Exception as e:
                        print(f"Health check error for {server_id}: {e}")

                    # Wait before next check
                    await asyncio.sleep(check_interval)

            async def check_health(self, server: Dict) -> bool:
                """
                Perform HTTP health check

                Args:
                    server: Server configuration

                Returns:
                    True if healthy, False otherwise
                """
                url = f"http://{server['ip']}:{server['port']}{server['health_check_path']}"

                try:
                    async with aiohttp.ClientSession() as session:
                        start_time = time.time()

                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=self.timeout)
                        ) as response:
                            latency = (time.time() - start_time) * 1000  # ms

                            # Check status code
                            if response.status != 200:
                                print(f"Server {server['id']} unhealthy: HTTP {response.status}")
                                return False

                            # Parse response
                            data = await response.json()

                            # Validate health response
                            if data.get('status') != 'healthy':
                                print(f"Server {server['id']} reports unhealthy: {data}")
                                return False

                            # Store latency metric
                            await self.redis.zadd(
                                f"server:{server['id']}:latency",
                                {str(time.time()): latency}
                            )

                            print(f"Server {server['id']} healthy (latency: {latency:.1f}ms)")
                            return True

                except asyncio.TimeoutError:
                    print(f"Server {server['id']} unhealthy: timeout")
                    return False
                except aiohttp.ClientError as e:
                    print(f"Server {server['id']} unhealthy: {e}")
                    return False

            async def update_health_state(self, server_id: str, is_healthy: bool):
                """
                Update server health state with threshold logic

                Args:
                    server_id: Server identifier
                    is_healthy: Current health check result
                """
                # Get current state
                state = await self.get_server_state(server_id)

                if is_healthy:
                    state['consecutive_successes'] += 1
                    state['consecutive_failures'] = 0

                    # Mark healthy after N consecutive successes
                    if state['consecutive_successes'] >= self.healthy_threshold:
                        if state['status'] != HealthStatus.HEALTHY:
                            print(f"✅ Server {server_id} marked HEALTHY")
                            state['status'] = HealthStatus.HEALTHY
                            await self.notify_load_balancer(server_id, HealthStatus.HEALTHY)
                else:
                    state['consecutive_failures'] += 1
                    state['consecutive_successes'] = 0

                    # Mark unhealthy after N consecutive failures
                    if state['consecutive_failures'] >= self.unhealthy_threshold:
                        if state['status'] != HealthStatus.UNHEALTHY:
                            print(f"❌ Server {server_id} marked UNHEALTHY")
                            state['status'] = HealthStatus.UNHEALTHY
                            await self.notify_load_balancer(server_id, HealthStatus.UNHEALTHY)

                # Persist state
                await self.save_server_state(server_id, state)

            async def get_server_state(self, server_id: str) -> Dict:
                """Get server state from Redis"""
                state = await self.redis.hgetall(f"server:{server_id}")

                if not state:
                    return {
                        'status': HealthStatus.UNKNOWN,
                        'consecutive_successes': 0,
                        'consecutive_failures': 0,
                        'last_check': 0
                    }

                return {
                    'status': HealthStatus(state.get('status', 'unknown')),
                    'consecutive_successes': int(state.get('consecutive_successes', 0)),
                    'consecutive_failures': int(state.get('consecutive_failures', 0)),
                    'last_check': float(state.get('last_check', 0))
                }

            async def save_server_state(self, server_id: str, state: Dict):
                """Save server state to Redis"""
                await self.redis.hmset(f"server:{server_id}", {
                    'status': state['status'].value,
                    'consecutive_successes': state['consecutive_successes'],
                    'consecutive_failures': state['consecutive_failures'],
                    'last_check': time.time()
                })

            async def notify_load_balancer(self, server_id: str, status: HealthStatus):
                """Notify load balancer of status change"""
                await self.redis.publish('health_updates', {
                    'server_id': server_id,
                    'status': status.value,
                    'timestamp': time.time()
                })
        ```

        ---

        ### Passive Health Checks

        **Monitor real client requests for failures.**

        ```python
        class PassiveHealthChecker:
            """
            Passive health monitoring based on real request failures
            """

            def __init__(self, redis_client):
                self.redis = redis_client
                self.failure_window = 60  # seconds
                self.failure_threshold = 5  # failures per minute

            async def record_request(self, server_id: str, success: bool, latency_ms: float):
                """
                Record request result

                Args:
                    server_id: Backend server ID
                    success: True if request succeeded
                    latency_ms: Request latency in milliseconds
                """
                timestamp = time.time()

                # Record in time-series
                metric_key = f"server:{server_id}:requests"
                await self.redis.zadd(metric_key, {
                    f"{timestamp}:{success}:{latency_ms}": timestamp
                })

                # Cleanup old data (older than window)
                cutoff = timestamp - self.failure_window
                await self.redis.zremrangebyscore(metric_key, '-inf', cutoff)

                # Check failure rate
                if not success:
                    failure_count = await self.get_recent_failure_count(server_id)

                    if failure_count >= self.failure_threshold:
                        print(f"⚠️ Server {server_id} exceeds failure threshold: {failure_count}")
                        await self.mark_unhealthy(server_id)

            async def get_recent_failure_count(self, server_id: str) -> int:
                """Count failures in recent window"""
                metric_key = f"server:{server_id}:requests"

                # Get all records in window
                records = await self.redis.zrange(metric_key, 0, -1)

                # Count failures
                failure_count = sum(
                    1 for record in records
                    if ':False:' in record.decode()
                )

                return failure_count

            async def mark_unhealthy(self, server_id: str):
                """Mark server as unhealthy based on passive checks"""
                await self.redis.hset(f"server:{server_id}", 'passive_status', 'unhealthy')
                await self.redis.publish('health_updates', {
                    'server_id': server_id,
                    'status': 'unhealthy',
                    'source': 'passive',
                    'timestamp': time.time()
                })
        ```

        ---

        ## Health Check Strategy

        **Combine active and passive for best results:**

        | Check Type | Pros | Cons | Use Case |
        |------------|------|------|----------|
        | **Active only** | Proactive, detects before client impact | Overhead, false positives (network blips) | Low-traffic services |
        | **Passive only** | No overhead, reflects real traffic | Reactive, slow detection | High-traffic services |
        | **Active + Passive** | Fast detection, low false positives | More complex | **Production (recommended)** |

        **Recommended configuration:**

        ```yaml
        health_check:
          active:
            enabled: true
            interval: 5s
            timeout: 2s
            path: /health
            healthy_threshold: 3
            unhealthy_threshold: 2

          passive:
            enabled: true
            failure_window: 60s
            failure_threshold: 5

          exponential_backoff:
            enabled: true
            max_interval: 60s
        ```

    === "🔒 SSL Termination"

        ## The Challenge

        **Problem:** SSL/TLS encryption/decryption is CPU-intensive. At 1M req/s with SSL, it consumes 500+ CPU cores.

        **SSL handshake cost:**

        ```
        RSA 2048-bit handshake: ~2-3ms CPU time
        ECDSA handshake: ~0.5-1ms CPU time

        At 1M req/s:
        - RSA: 1M × 2ms = 2000 CPU seconds/sec = 2000 cores!
        - ECDSA: 1M × 0.5ms = 500 cores
        ```

        ---

        ## SSL Termination Architecture

        **Why terminate SSL at load balancer:**

        1. **Offload backend CPU** - Backends handle plaintext HTTP
        2. **Centralized certificates** - One place to manage certs
        3. **Enable L7 features** - Content inspection requires plaintext
        4. **Connection reuse** - Keep-alive to backends

        **Architecture:**

        ```
        Client ---HTTPS---> Load Balancer ---HTTP---> Backend
              (encrypted)    (decrypt here)    (plaintext)
        ```

        ---

        ## SSL Optimization Techniques

        ### 1. Hardware Acceleration

        **Use dedicated SSL acceleration hardware:**

        ```python
        # OpenSSL with AES-NI hardware acceleration
        import ssl
        import socket

        def create_ssl_context():
            """Create SSL context with hardware acceleration"""
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

            # Load certificates
            context.load_cert_chain(
                certfile="/etc/ssl/cert.pem",
                keyfile="/etc/ssl/key.pem"
            )

            # Enable modern protocols only
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3

            # Use secure ciphers (hardware-accelerated)
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS')

            # Enable session tickets (reduces handshakes)
            context.options |= ssl.OP_NO_TICKET  # Or manage custom tickets

            return context
        ```

        **Benefits:**

        - **10x speedup** - Hardware AES-NI encryption
        - **Lower CPU** - Offload to dedicated crypto engines
        - **Higher throughput** - 100k+ req/s per server

        ---

        ### 2. Session Resumption

        **Reuse SSL session to skip handshake:**

        ```python
        class SSLSessionCache:
            """
            SSL session resumption cache
            Reduces handshake overhead by ~80%
            """

            def __init__(self, redis_client):
                self.redis = redis_client
                self.ttl = 3600  # 1 hour

            async def get_session(self, session_id: str) -> bytes:
                """
                Retrieve cached SSL session

                Args:
                    session_id: SSL session identifier

                Returns:
                    Serialized SSL session data
                """
                session_data = await self.redis.get(f"ssl:session:{session_id}")
                return session_data if session_data else None

            async def store_session(self, session_id: str, session_data: bytes):
                """
                Store SSL session for resumption

                Args:
                    session_id: SSL session identifier
                    session_data: Serialized session data
                """
                await self.redis.setex(
                    f"ssl:session:{session_id}",
                    self.ttl,
                    session_data
                )
        ```

        **Impact:**

        ```
        Without session resumption:
        - Every request: Full handshake (2ms)
        - 1M req/s × 2ms = 2000 CPU cores

        With session resumption (80% cache hit):
        - 80% requests: Session reuse (0.1ms)
        - 20% requests: Full handshake (2ms)
        - 1M × (0.8 × 0.1ms + 0.2 × 2ms) = 480 CPU cores

        Savings: 76% reduction in CPU usage!
        ```

        ---

        ### 3. TLS 1.3 (Modern)

        **TLS 1.3 improvements:**

        - **1-RTT handshake** - Faster connection (vs 2-RTT in TLS 1.2)
        - **0-RTT resumption** - Zero round-trip for resumed sessions
        - **Stronger ciphers** - ChaCha20, AES-GCM only

        **Configuration:**

        ```nginx
        # NGINX SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
        ssl_prefer_server_ciphers on;

        # Session resumption
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        ssl_session_tickets on;

        # OCSP stapling (faster cert validation)
        ssl_stapling on;
        ssl_stapling_verify on;

        # Hardware acceleration
        ssl_engine aesni;
        ```

        ---

        ### 4. Certificate Selection

        **ECDSA vs RSA:**

        | Cert Type | Key Size | Handshake Time | Security Level |
        |-----------|----------|----------------|----------------|
        | **RSA** | 2048-bit | 2-3ms | Good |
        | **RSA** | 4096-bit | 8-10ms | Better |
        | **ECDSA** | 256-bit | 0.5-1ms | Equivalent to RSA 3072 |

        **Recommendation:** Use **ECDSA P-256** for 4x faster handshakes with equivalent security.

        ---

        ## End-to-End SSL (Alternative)

        **For sensitive data, use end-to-end encryption:**

        ```
        Client ---HTTPS---> Load Balancer ---HTTPS---> Backend
              (encrypted)  (pass-through)    (still encrypted)
        ```

        **Trade-offs:**

        | Aspect | SSL Termination | End-to-End SSL |
        |--------|-----------------|----------------|
        | **Backend CPU** | ✅ Low (plaintext) | ❌ High (decrypt) |
        | **L7 features** | ✅ Available (inspect) | ❌ Limited (no inspect) |
        | **Security** | 🟡 LB sees plaintext | ✅ End-to-end encryption |
        | **Certificate mgmt** | ✅ Centralized | ❌ Distributed |
        | **Performance** | ✅ Fast | 🟡 Slower |

        **Recommendation:**

        - **SSL termination** - Most use cases (99%)
        - **End-to-end SSL** - Financial, healthcare, PCI compliance

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling load balancer from 10k to 1M requests/sec.

    **Scaling challenges at 1M req/s:**

    - **Network bandwidth:** 400 Gbps egress
    - **Connection rate:** 10k new connections/sec
    - **SSL handshakes:** 200k handshakes/sec (20% new)
    - **Health checks:** 10k checks/sec (1000 servers × 10 LB)

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Network I/O** | ✅ Yes (400 Gbps) | Multiple NICs, NIC teaming, 100 Gbps NICs |
    | **SSL/TLS** | ✅ Yes (500 cores) | Hardware acceleration, TLS 1.3, ECDSA certs |
    | **Connection tracking** | ✅ Yes (6.4 GB RAM) | Connection pooling, aggressive timeouts |
    | **Health checks** | 🟡 Approaching | Reduce frequency, passive checks |
    | **Session store (Redis)** | 🟡 Approaching | Redis cluster, read replicas |

    ---

    ## Horizontal Scaling

    ### DNS-based Load Balancing (First Tier)

    ```
    Client query: api.example.com

    DNS response (round-robin):
    - 203.0.113.10 (LB-1)
    - 203.0.113.11 (LB-2)
    - 203.0.113.12 (LB-3)
    ...
    - 203.0.113.29 (LB-20)

    Benefits:
    - Distribute across 20 LB instances
    - Each LB: 50k req/s (manageable)
    - No single point of failure
    ```

    **DNS configuration:**

    ```dns
    api.example.com.  60  IN  A  203.0.113.10
    api.example.com.  60  IN  A  203.0.113.11
    api.example.com.  60  IN  A  203.0.113.12
    ...

    TTL: 60 seconds (fast failover)
    Health-aware: Remove failed LBs from DNS
    ```

    ---

    ### Anycast Routing (Advanced)

    **Single IP, multiple geographic locations:**

    ```
    Single VIP: 203.0.113.100

    BGP Anycast routing:
    - US-East: LB cluster (10 instances)
    - US-West: LB cluster (10 instances)
    - EU: LB cluster (10 instances)
    - Asia: LB cluster (10 instances)

    Client automatically routed to nearest location
    ```

    **Benefits:**

    - **Low latency** - Route to nearest data center
    - **High availability** - Automatic failover across regions
    - **DDoS protection** - Traffic distributed globally

    ---

    ## Vertical Scaling Limits

    **Single LB instance limits:**

    | Resource | Limit | Workaround |
    |----------|-------|------------|
    | **Network bandwidth** | 25 Gbps (AWS m5.24xlarge) | Multiple NICs, bond interfaces |
    | **CPU cores** | 96 cores (m5.24xlarge) | Hardware acceleration, offload SSL |
    | **Memory** | 384 GB (m5.24xlarge) | Reduce connection buffers, aggressive timeouts |
    | **Network PPS** | 10M packets/sec | Kernel bypass (DPDK), XDP |
    | **Concurrent connections** | 1M connections | Increase ulimit, kernel tuning |

    **Recommendation:** Scale horizontally beyond 100k req/s per instance.

    ---

    ## Performance Optimization

    ### 1. Connection Pooling to Backends

    **Reuse backend connections:**

    ```python
    class BackendConnectionPool:
        """
        Connection pool to backend servers
        Reduces overhead of new connections
        """

        def __init__(self, backend_ip, backend_port, pool_size=100):
            self.backend_ip = backend_ip
            self.backend_port = backend_port
            self.pool_size = pool_size
            self.pool = asyncio.Queue(maxsize=pool_size)

            # Pre-create connections
            for _ in range(pool_size):
                conn = self._create_connection()
                self.pool.put_nowait(conn)

        def _create_connection(self):
            """Create new backend connection"""
            return socket.create_connection(
                (self.backend_ip, self.backend_port),
                timeout=5
            )

        async def get_connection(self):
            """Get connection from pool"""
            return await self.pool.get()

        async def return_connection(self, conn):
            """Return connection to pool"""
            if self._is_valid(conn):
                await self.pool.put(conn)
            else:
                conn.close()
                new_conn = self._create_connection()
                await self.pool.put(new_conn)
    ```

    **Impact:**

    ```
    Without pooling:
    - New connection per request: 3-way handshake (1ms)
    - 1M req/s × 1ms = 1000 CPU cores

    With pooling (100 connections per backend):
    - Connection reuse: ~0 ms overhead
    - Only 10k new connections/sec (connection churn)
    - 10k × 1ms = 10 CPU cores

    Savings: 99% reduction!
    ```

    ---

    ### 2. Kernel Tuning (Linux)

    **Optimize kernel for high throughput:**

    ```bash
    # /etc/sysctl.conf

    # Increase connection tracking table size
    net.netfilter.nf_conntrack_max = 2000000
    net.netfilter.nf_conntrack_buckets = 500000

    # TCP tuning
    net.ipv4.tcp_max_syn_backlog = 8192
    net.core.somaxconn = 8192
    net.ipv4.tcp_fin_timeout = 15
    net.ipv4.tcp_tw_reuse = 1
    net.ipv4.tcp_keepalive_time = 300

    # Increase file descriptor limit
    fs.file-max = 2000000

    # Buffer sizes
    net.core.rmem_max = 16777216
    net.core.wmem_max = 16777216
    net.ipv4.tcp_rmem = 4096 87380 16777216
    net.ipv4.tcp_wmem = 4096 65536 16777216

    # Enable BBR congestion control (better throughput)
    net.core.default_qdisc = fq
    net.ipv4.tcp_congestion_control = bbr
    ```

    ---

    ### 3. Zero-Copy Networking

    **Use sendfile() to avoid copying data:**

    ```c
    // Traditional approach: 4 copies
    read(file_fd, buffer, size);      // Copy 1: disk -> kernel
                                       // Copy 2: kernel -> user space
    write(socket_fd, buffer, size);   // Copy 3: user space -> kernel
                                       // Copy 4: kernel -> NIC

    // Zero-copy: 2 copies (bypass user space)
    sendfile(socket_fd, file_fd, NULL, size);  // disk -> kernel -> NIC
    ```

    **Impact:** 50% reduction in CPU usage for static content.

    ---

    ## Cost Optimization

    **Monthly cost at 1M req/s:**

    | Component | Quantity | Cost |
    |-----------|----------|------|
    | **Load Balancer (c5.4xlarge)** | 20 instances | $24,000 |
    | **SSL Acceleration** | Hardware | Included in instance |
    | **Elastic IPs** | 20 IPs | $100 |
    | **Network egress** | 1.1 PB/month | $110,000 |
    | **Redis (session store)** | 3-node cluster | $1,500 |
    | **Monitoring (CloudWatch)** | Metrics, logs | $500 |
    | **Route 53 (DNS)** | Queries | $200 |
    | **Total** | - | **$136,300/month** |

    **Cost optimization strategies:**

    1. **Use CDN** - Cache static content, reduce egress (save $80k/month)
    2. **Reserved instances** - 40% discount on LB instances (save $10k/month)
    3. **Compress responses** - Reduce bandwidth by 60% (save $20k/month)
    4. **Regional load balancing** - Route traffic locally (reduce latency tax)

    **Optimized cost:** ~$60k/month (56% savings)

    ---

    ## Monitoring & Alerting

    **Key metrics to monitor:**

    | Metric | Target | Alert Threshold | Action |
    |--------|--------|-----------------|--------|
    | **Request latency (P95)** | < 1ms | > 5ms | Scale LB, check backends |
    | **Error rate** | < 0.1% | > 1% | Check backend health, failover |
    | **Active connections** | 50k | > 90k | Add LB capacity |
    | **SSL handshake rate** | 10k/s | > 30k/s | Add LB, enable session cache |
    | **Backend health** | 95%+ | < 90% | Investigate backend issues |
    | **Network throughput** | 20 Gbps | > 22 Gbps | Add NICs, horizontal scaling |
    | **CPU utilization** | < 60% | > 80% | Add instances, optimize code |

    **Prometheus queries:**

    ```promql
    # P95 latency
    histogram_quantile(0.95, rate(lb_request_duration_seconds_bucket[5m]))

    # Error rate
    rate(lb_requests_total{status=~"5.."}[5m]) / rate(lb_requests_total[5m])

    # Backend health percentage
    sum(lb_backend_healthy) / sum(lb_backend_total) * 100

    # Request throughput
    rate(lb_requests_total[5m])
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Hybrid L4/L7** - Layer 4 for speed, Layer 7 for features
    2. **Active + passive health checks** - Fast failure detection, low false positives
    3. **Consistent hashing** - Minimal session disruption during scaling
    4. **SSL termination at LB** - Offload backend CPU, enable L7 features
    5. **Hardware acceleration** - Dedicated SSL engines, NIC offload
    6. **Connection pooling** - Reuse backend connections, reduce overhead
    7. **Horizontal scaling** - DNS round-robin, Anycast for global distribution

    ---

    ## Interview Tips

    ✅ **Discuss algorithm trade-offs** - Round robin vs least connections vs consistent hashing

    ✅ **Explain health check strategies** - Active vs passive, thresholds, backoff

    ✅ **Clarify L4 vs L7** - Performance vs features, when to use each

    ✅ **Address SSL overhead** - Hardware acceleration, session resumption, TLS 1.3

    ✅ **Talk about scaling limits** - Network I/O, CPU, connection limits

    ✅ **Mention real-world solutions** - HAProxy, NGINX, AWS ELB, Envoy

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle backend server failures?"** | Active + passive health checks, automatic removal, gradual re-addition |
    | **"How to maintain sessions during scaling?"** | Consistent hashing with virtual nodes, sticky sessions via cookies/Redis |
    | **"Layer 4 vs Layer 7 - which to use?"** | L4 for low latency (<0.1ms), L7 for content routing and SSL termination |
    | **"How to optimize SSL performance?"** | Hardware acceleration, session resumption, TLS 1.3, ECDSA certificates |
    | **"How to prevent single point of failure?"** | Multiple LB instances, DNS round-robin, Anycast, active-active deployment |
    | **"How to handle 1M requests/sec?"** | Horizontal scaling (20+ LB instances), hardware acceleration, kernel tuning |

    ---

    ## Load Balancer Comparison

    | Product | Type | Max Throughput | Use Case |
    |---------|------|----------------|----------|
    | **HAProxy** | Software (L4/L7) | 1M+ req/s | General purpose, high performance |
    | **NGINX** | Software (L7) | 500k req/s | HTTP/HTTPS, reverse proxy, caching |
    | **AWS ELB** | Managed (L4/L7) | Auto-scaling | AWS cloud, zero maintenance |
    | **Envoy** | Software (L7) | 300k req/s | Service mesh, microservices |
    | **F5 BIG-IP** | Hardware (L4/L7) | 10M+ req/s | Enterprise, very high throughput |
    | **Cloudflare** | Global (L7) | Unlimited | Global CDN + load balancing |

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** HAProxy, NGINX, AWS, F5, Cloudflare, Google, Meta, Netflix

---

*Master this problem and you'll be ready for: API Gateway, Reverse Proxy, CDN, Service Mesh, Traffic Management*
