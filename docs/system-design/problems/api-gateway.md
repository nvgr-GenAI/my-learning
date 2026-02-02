# Design API Gateway

A centralized entry point that acts as a reverse proxy, routing client requests to appropriate backend microservices while handling cross-cutting concerns like authentication, rate limiting, request transformation, and load balancing.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 500K requests/sec, 10K microservices behind gateway, 100M daily active users |
| **Key Challenges** | Low latency routing (<10ms overhead), authentication/authorization, rate limiting, circuit breaker, service discovery |
| **Core Concepts** | Reverse proxy, JWT validation, token bucket rate limiting, circuit breaker pattern, request/response transformation |
| **Companies** | Kong, AWS (API Gateway), Google (Apigee), Netflix (Zuul), Kong, Cloudflare, Azure API Management |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Request Routing** | Route requests to appropriate backend services | P0 (Must have) |
    | **Authentication** | JWT/OAuth token validation | P0 (Must have) |
    | **Rate Limiting** | Limit requests per user/API key | P0 (Must have) |
    | **Load Balancing** | Distribute traffic across service instances | P0 (Must have) |
    | **Request/Response Transformation** | Modify headers, body, query params | P1 (Should have) |
    | **Circuit Breaker** | Prevent cascading failures | P1 (Should have) |
    | **Request Aggregation** | Combine multiple backend calls | P1 (Should have) |
    | **Caching** | Cache responses for GET requests | P1 (Should have) |
    | **API Analytics** | Track request counts, latency, errors | P2 (Nice to have) |
    | **API Versioning** | Support multiple API versions | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Service mesh implementation (Istio/Linkerd)
    - GraphQL federation
    - WebSocket proxying (real-time)
    - Complex ML-based anomaly detection
    - Advanced security (DDoS protection, WAF)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency Overhead** | < 10ms p95 | Gateway shouldn't add significant latency |
    | **Availability** | 99.99% uptime | Gateway is single point of failure |
    | **Throughput** | 500K requests/sec | Must handle massive concurrent traffic |
    | **Scalability** | Horizontal scaling | Add nodes to handle traffic spikes |
    | **Security** | Zero trust architecture | All requests authenticated/authorized |
    | **Consistency** | Eventual consistency for config | Config changes propagate within 5 seconds |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users: 100M
    Average requests per user: 50 requests/day

    Total daily requests:
    - Daily requests: 100M × 50 = 5B requests/day
    - Requests per second: 5B / 86,400 = ~58,000 req/sec
    - Peak QPS (3x average): ~174,000 req/sec
    - Design for: 500K req/sec (3x peak for headroom)

    Request types:
    - GET (read): 70% = 350K req/sec
    - POST/PUT/DELETE (write): 30% = 150K req/sec

    Cache hit rate:
    - GET requests cacheable: 50%
    - Cache hit rate: 40%
    - Actual backend requests: 350K × 0.6 = 210K + 150K = 360K req/sec

    Authentication:
    - JWT tokens: 90% of requests
    - API keys: 10% of requests
    - Token validation: < 1ms (cached public keys)

    Rate limiting checks:
    - 100% of requests checked
    - Distributed rate limiter: Redis counters
    - Rate limit overhead: < 2ms
    ```

    ### Storage Estimates

    ```
    Configuration data:
    - Routes: 10K microservices × 5 endpoints = 50K routes × 2 KB = 100 MB
    - Rate limit rules: 1M users × 100 bytes = 100 MB
    - API keys: 1M API keys × 500 bytes = 500 MB
    - Total config: ~1 GB (fully cached in memory)

    Cache storage (response caching):
    - Cache 1% of GET requests
    - Average response size: 10 KB
    - Requests to cache: 350K req/sec × 0.01 = 3,500 req/sec
    - TTL: 60 seconds average
    - Total cached: 3,500 × 60 × 10 KB = 2.1 GB
    - Cache for top 10% hot data: 10 GB Redis cluster

    Analytics data (metrics):
    - Request logs: 500K req/sec × 1 KB = 500 MB/sec = 43 TB/day
    - Aggregated metrics: 100 GB/day (time-series DB)
    - Retention: 30 days = 3 TB

    JWT token cache:
    - Active users: 10M concurrent × 2 KB = 20 GB
    - Parsed JWT claims cached for 5 minutes
    ```

    ### Bandwidth Estimates

    ```
    Request ingress:
    - Average request size: 5 KB (headers + body)
    - 500K req/sec × 5 KB = 2.5 GB/sec = 20 Gbps

    Response egress:
    - Average response size: 10 KB
    - 500K req/sec × 10 KB = 5 GB/sec = 40 Gbps

    Total bandwidth: 20 Gbps (ingress) + 40 Gbps (egress) = 60 Gbps
    ```

    ### Memory Estimates (Per Gateway Node)

    ```
    Configuration:
    - Route rules: 100 MB
    - Rate limit config: 100 MB
    - Circuit breaker state: 50 MB

    Request state:
    - Connection pooling: 10K connections × 100 KB = 1 GB
    - Request buffering: 100 requests × 5 KB = 500 KB

    Caching:
    - Response cache (local): 1 GB
    - JWT token cache: 2 GB
    - Rate limit counters (local): 500 MB

    Total per node: ~5 GB RAM
    Nodes needed: 100 nodes (5K req/sec each)
    ```

    ---

    ## Key Assumptions

    1. JWT tokens valid for 1 hour, refresh every 15 minutes
    2. 90% of traffic is authenticated (JWT/API key)
    3. Rate limits: 1000 req/min per user, 10K req/min per API key
    4. 40% cache hit rate for GET requests
    5. Circuit breaker: 50% error rate triggers open, 30s timeout
    6. Service discovery updates propagate within 5 seconds

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Single entry point:** All client requests flow through gateway
    2. **Zero trust:** Authenticate and authorize every request
    3. **Fail-safe:** Circuit breaker prevents cascading failures
    4. **Stateless:** No session state in gateway (JWT tokens)
    5. **Horizontally scalable:** Add nodes to increase throughput
    6. **Configuration-driven:** Dynamic routing rules (no code deploy)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile Apps]
            Web[Web Apps]
            ThirdParty[Third Party APIs]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets]
            WAF[WAF<br/>DDoS protection]
        end

        subgraph "API Gateway Cluster"
            LB[Load Balancer<br/>L7 load balancing]
            GW1[Gateway Node 1<br/>Routing, Auth, Rate Limit]
            GW2[Gateway Node 2<br/>Routing, Auth, Rate Limit]
            GW3[Gateway Node N<br/>Routing, Auth, Rate Limit]
        end

        subgraph "Gateway Services"
            Auth[Auth Service<br/>JWT validation]
            RateLimit[Rate Limit Service<br/>Distributed counters]
            Discovery[Service Discovery<br/>Consul/Eureka]
            Config[Config Service<br/>Dynamic routing]
        end

        subgraph "Caching & State"
            Redis[Redis Cluster<br/>Rate limiting, caching]
            Cache[Response Cache<br/>Hot data]
        end

        subgraph "Backend Microservices"
            Users[User Service<br/>3 instances]
            Orders[Order Service<br/>5 instances]
            Products[Product Service<br/>4 instances]
            Payments[Payment Service<br/>3 instances]
            Analytics[Analytics Service<br/>2 instances]
        end

        subgraph "Monitoring"
            Metrics[Metrics<br/>Prometheus]
            Logs[Logs<br/>ELK Stack]
            Tracing[Tracing<br/>Jaeger]
        end

        Mobile --> CDN
        Web --> CDN
        ThirdParty --> WAF

        CDN --> LB
        WAF --> LB

        LB --> GW1
        LB --> GW2
        LB --> GW3

        GW1 --> Auth
        GW1 --> RateLimit
        GW1 --> Discovery
        GW1 --> Config

        GW2 --> Auth
        GW2 --> RateLimit
        GW2 --> Discovery
        GW2 --> Config

        GW3 --> Auth
        GW3 --> RateLimit
        GW3 --> Discovery
        GW3 --> Config

        Auth --> Redis
        RateLimit --> Redis
        Discovery --> Config

        GW1 --> Cache
        GW2 --> Cache
        GW3 --> Cache

        GW1 --> Users
        GW1 --> Orders
        GW1 --> Products
        GW1 --> Payments
        GW1 --> Analytics

        GW2 --> Users
        GW2 --> Orders
        GW2 --> Products
        GW2 --> Payments

        GW3 --> Users
        GW3 --> Orders

        Users --> Metrics
        Orders --> Metrics
        Products --> Metrics

        GW1 --> Logs
        GW2 --> Logs
        GW3 --> Logs

        GW1 --> Tracing
        GW2 --> Tracing
        GW3 --> Tracing

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis fill:#fff4e1
        style Cache fill:#fff4e1
        style Users fill:#ffe1e1
        style Orders fill:#ffe1e1
        style Products fill:#ffe1e1
        style Payments fill:#ffe1e1
        style Metrics fill:#e8eaf6
        style Logs fill:#e8eaf6
        style Tracing fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis Cluster** | Distributed rate limiting (1M ops/sec), token cache, shared state | Memcached (no distributed counters), local memory (no sharing) |
    | **Service Discovery** | Dynamic service instances, health checks, load balancing | Hardcoded IPs (no auto-scaling), DNS (slow updates, no health) |
    | **Circuit Breaker** | Prevent cascading failures, fast fail, self-healing | Retry only (amplifies failures), timeouts (still waits) |
    | **JWT** | Stateless auth, no session store, claims-based | Session cookies (stateful, DB lookup), API keys only (no expiry) |
    | **Connection Pooling** | Reuse TCP connections, reduce latency (3-way handshake) | New connection per request (slow, resource intensive) |
    | **Config Service** | Dynamic routing, no code deploy, A/B testing | Hardcoded routes (requires deploy), file-based (no hot reload) |

    **Key Trade-off:** We chose **availability over consistency**. Rate limiting counters may have brief inconsistencies during network partitions, but gateway remains available.

    ---

    ## API Design

    ### Gateway Configuration API

    **1. Create Route**

    **Request:**
    ```http
    POST /api/v1/admin/routes
    Content-Type: application/json
    Authorization: Bearer <admin_token>

    {
      "name": "user-service-route",
      "path": "/api/users/*",
      "methods": ["GET", "POST", "PUT", "DELETE"],
      "upstream": {
        "service_name": "user-service",
        "load_balancer": "round_robin",
        "health_check": {
          "path": "/health",
          "interval": 10,
          "timeout": 5
        }
      },
      "plugins": [
        {
          "name": "jwt-auth",
          "config": {
            "key_claim": "sub",
            "secret": "env://JWT_SECRET"
          }
        },
        {
          "name": "rate-limit",
          "config": {
            "limit": 1000,
            "window": 60,
            "key": "jwt.sub"
          }
        },
        {
          "name": "circuit-breaker",
          "config": {
            "error_threshold": 50,
            "timeout": 30
          }
        }
      ],
      "timeout": 30,
      "retries": 2
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "route_id": "route_abc123",
      "name": "user-service-route",
      "created_at": "2026-02-02T10:30:00Z",
      "status": "active"
    }
    ```

    ---

    ### 2. Proxy Request (Client-facing)

    **Request:**
    ```http
    GET /api/users/12345 HTTP/1.1
    Host: api.example.com
    Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    X-API-Key: api_key_xyz789
    Accept: application/json
    ```

    **Gateway Processing:**
    ```
    1. Rate limit check (Redis)
    2. JWT validation (parse + verify signature)
    3. Route matching (/api/users/* → user-service)
    4. Service discovery (get healthy instances)
    5. Load balancing (select instance)
    6. Circuit breaker check (is circuit open?)
    7. Request transformation (add headers)
    8. Forward to backend
    9. Response transformation
    10. Cache response (if cacheable)
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json
    X-RateLimit-Remaining: 842
    X-RateLimit-Reset: 1643732400
    X-Gateway-Latency: 8ms

    {
      "user_id": "12345",
      "username": "john_doe",
      "email": "john@example.com"
    }
    ```

    ---

    ### 3. Rate Limit Status

    **Request:**
    ```http
    GET /api/v1/rate-limit/status
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user123",
      "limits": [
        {
          "route": "/api/users/*",
          "limit": 1000,
          "window": 60,
          "remaining": 842,
          "reset_at": "2026-02-02T10:31:00Z"
        },
        {
          "route": "/api/orders/*",
          "limit": 500,
          "window": 60,
          "remaining": 489,
          "reset_at": "2026-02-02T10:31:00Z"
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Routes (PostgreSQL)

    ```sql
    -- Routes configuration
    CREATE TABLE routes (
        route_id VARCHAR(36) PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        path VARCHAR(500) NOT NULL,
        methods TEXT[], -- ['GET', 'POST', etc.]
        upstream_service VARCHAR(255) NOT NULL,
        load_balancer_strategy VARCHAR(50) DEFAULT 'round_robin',
        timeout_seconds INT DEFAULT 30,
        retries INT DEFAULT 2,
        config JSONB, -- Plugin configs, transformations, etc.
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'active',
        INDEX idx_path (path),
        INDEX idx_status (status)
    );

    -- Service instances (from service discovery)
    CREATE TABLE service_instances (
        instance_id VARCHAR(36) PRIMARY KEY,
        service_name VARCHAR(255) NOT NULL,
        host VARCHAR(255) NOT NULL,
        port INT NOT NULL,
        health_status VARCHAR(20) DEFAULT 'healthy',
        last_health_check TIMESTAMP,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_service (service_name),
        INDEX idx_health (health_status)
    );

    -- API Keys
    CREATE TABLE api_keys (
        key_id VARCHAR(36) PRIMARY KEY,
        api_key VARCHAR(64) UNIQUE NOT NULL,
        user_id VARCHAR(36) NOT NULL,
        name VARCHAR(255),
        scopes TEXT[], -- ['read:users', 'write:orders']
        rate_limit INT DEFAULT 10000, -- per hour
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used_at TIMESTAMP,
        revoked BOOLEAN DEFAULT FALSE,
        INDEX idx_api_key (api_key),
        INDEX idx_user (user_id)
    );
    ```

    ---

    ### Rate Limiting (Redis)

    ```python
    # Rate limit counters (sliding window)
    # Key format: ratelimit:{route_id}:{user_id}:{timestamp_bucket}
    # Example: ratelimit:route123:user456:1643732400

    redis_keys = {
        # Token bucket
        f"ratelimit:{route_id}:{user_id}:tokens": 950,  # Remaining tokens
        f"ratelimit:{route_id}:{user_id}:last_refill": 1643732385,  # Last refill time

        # Sliding window log (for accurate counting)
        f"ratelimit:{route_id}:{user_id}:requests": [
            # Sorted set: score = timestamp, value = request_id
            (1643732380, "req1"),
            (1643732385, "req2"),
            (1643732390, "req3"),
            # ...
        ],

        # Circuit breaker state
        f"circuit:{service_name}:state": "closed",  # closed, open, half_open
        f"circuit:{service_name}:failures": 45,
        f"circuit:{service_name}:last_failure": 1643732390,
        f"circuit:{service_name}:opened_at": None,
    }
    ```

    ---

    ## Data Flow Diagrams

    ### Request Processing Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway
        participant Redis
        participant Auth
        participant Discovery
        participant Backend
        participant Cache

        Client->>Gateway: GET /api/users/123<br/>Authorization: Bearer <jwt>

        Note over Gateway: 1. Rate Limiting
        Gateway->>Redis: INCR ratelimit:user123:bucket
        Redis-->>Gateway: counter=45, limit=1000

        alt Rate limit exceeded
            Gateway-->>Client: 429 Too Many Requests
        end

        Note over Gateway: 2. Authentication
        Gateway->>Gateway: Parse JWT header
        Gateway->>Cache: GET jwt:cache:<token_hash>

        alt Cache miss
            Cache-->>Gateway: null
            Gateway->>Auth: Validate JWT signature
            Auth-->>Gateway: Valid, claims={user_id:123}
            Gateway->>Cache: SET jwt:cache:<token_hash> TTL=300s
        else Cache hit
            Cache-->>Gateway: claims={user_id:123}
        end

        Note over Gateway: 3. Authorization
        Gateway->>Gateway: Check scopes/permissions

        Note over Gateway: 4. Route Matching
        Gateway->>Gateway: Match path /api/users/123<br/>→ route: user-service

        Note over Gateway: 5. Service Discovery
        Gateway->>Discovery: Get instances for user-service
        Discovery-->>Gateway: [instance1:8080, instance2:8080, instance3:8080]

        Note over Gateway: 6. Load Balancing
        Gateway->>Gateway: Round robin → instance2:8080

        Note over Gateway: 7. Circuit Breaker Check
        Gateway->>Redis: GET circuit:user-service:state
        Redis-->>Gateway: "closed" (healthy)

        alt Circuit open (too many failures)
            Gateway-->>Client: 503 Service Unavailable
        end

        Note over Gateway: 8. Check Response Cache
        Gateway->>Cache: GET cache:GET:/api/users/123

        alt Cache hit (40% of GET requests)
            Cache-->>Gateway: {user_id:123, name:"John"}
            Gateway-->>Client: 200 OK (cached)
        else Cache miss
            Cache-->>Gateway: null

            Note over Gateway: 9. Forward Request
            Gateway->>Backend: GET /users/123<br/>X-User-ID: 123<br/>X-Request-ID: req_xyz

            alt Backend success
                Backend-->>Gateway: 200 OK {user_id:123, name:"John"}

                Note over Gateway: 10. Cache Response
                Gateway->>Cache: SET cache:GET:/api/users/123<br/>TTL=60s

                Note over Gateway: 11. Record Success
                Gateway->>Redis: DECR circuit:user-service:failures

                Gateway-->>Client: 200 OK<br/>X-RateLimit-Remaining: 955<br/>X-Gateway-Latency: 8ms
            else Backend error (5xx)
                Backend-->>Gateway: 503 Service Unavailable

                Note over Gateway: 12. Record Failure
                Gateway->>Redis: INCR circuit:user-service:failures
                Gateway->>Redis: GET circuit:user-service:failures
                Redis-->>Gateway: failures=51 (threshold=50)

                Note over Gateway: 13. Open Circuit
                Gateway->>Redis: SET circuit:user-service:state "open"<br/>EXPIRE 30s

                Gateway-->>Client: 503 Service Unavailable<br/>X-Circuit-State: open
            end
        end
    ```

    **Flow Explanation:**

    1. **Rate limiting** - Check Redis counter (< 2ms)
    2. **Authentication** - Validate JWT with cached public keys (< 1ms)
    3. **Authorization** - Check user permissions/scopes
    4. **Route matching** - Find backend service using path/method
    5. **Service discovery** - Get healthy instances from registry
    6. **Load balancing** - Select instance (round robin/least connections)
    7. **Circuit breaker** - Check if service is healthy
    8. **Response cache** - Return cached response if available
    9. **Forward request** - Proxy to backend with added headers
    10. **Cache response** - Store cacheable responses
    11. **Record metrics** - Track success/failure rates

    ---

    ### Circuit Breaker State Machine

    ```mermaid
    stateDiagram-v2
        [*] --> Closed

        Closed --> Open: Error rate > 50%<br/>(e.g., 51 failures in 100 requests)
        Closed --> Closed: Success request<br/>(decrement failure counter)

        Open --> HalfOpen: Wait 30 seconds<br/>(timeout period)
        Open --> Open: All requests fail fast<br/>(don't call backend)

        HalfOpen --> Closed: Success rate > 80%<br/>(test requests succeed)
        HalfOpen --> Open: Failure detected<br/>(test request fails)

        note right of Closed
            Normal operation
            All requests forwarded
            Track failure rate
        end note

        note right of Open
            Circuit tripped
            Fail fast (no backend calls)
            Wait for timeout
        end note

        note right of HalfOpen
            Testing recovery
            Allow limited requests
            Monitor success rate
        end note
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical API Gateway subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Request Routing** | How to route 500K req/sec with <10ms latency? | Trie-based path matching, compiled route rules, connection pooling |
    | **Authentication & Authorization** | How to validate JWT tokens at scale? | Cached public keys, parsed token cache, asymmetric crypto (RS256) |
    | **Rate Limiting** | How to limit per-user requests across distributed gateways? | Token bucket with Redis, sliding window log, distributed counters |
    | **Circuit Breaker** | How to prevent cascading failures? | Adaptive failure threshold, exponential backoff, half-open testing |

    ---

    === "🚦 Request Routing"

        ## The Challenge

        **Problem:** Route 500K requests/sec to 10K microservices with <10ms latency. Pattern matching on URL paths must be fast.

        **Naive approach:** Loop through all routes and regex match. **Doesn't scale** (O(n) for 10K routes).

        **Requirements:**

        - **Fast matching:** O(log n) or better
        - **Pattern support:** `/api/users/:id`, `/api/orders/*/items`
        - **Priority routing:** Specific routes before wildcards
        - **Hot reloading:** Update routes without restart

        ---

        ## Trie-based Route Matching

        **Data Structure:**

        ```
        Route Trie (Prefix Tree):

        Root
        ├── api
        │   ├── users
        │   │   ├── :id (user-service)
        │   │   │   └── orders (order-service)
        │   │   ├── * (user-service)
        │   │   └── search (search-service)
        │   ├── orders
        │   │   ├── :id (order-service)
        │   │   └── * (order-service)
        │   └── products
        │       └── * (product-service)
        └── health (health-service)
        ```

        **Complexity:**

        - **Insertion:** O(m) where m = path segments
        - **Search:** O(m) where m = path segments
        - **Memory:** O(total segments across all routes)

        ---

        ## Implementation

        ```python
        from typing import Dict, List, Optional, Tuple
        import re

        class RouteNode:
            """Node in the route trie"""

            def __init__(self):
                self.children: Dict[str, RouteNode] = {}
                self.route: Optional[Route] = None
                self.is_param = False  # :id param
                self.is_wildcard = False  # * wildcard
                self.param_name: Optional[str] = None

        class Route:
            """Route configuration"""

            def __init__(self, path: str, methods: List[str], service: str, config: dict):
                self.path = path
                self.methods = methods
                self.service = service
                self.config = config
                self.priority = self._calculate_priority(path)

            def _calculate_priority(self, path: str) -> int:
                """Higher priority = more specific routes"""
                # Static segments: priority 3
                # Param segments (:id): priority 2
                # Wildcard segments (*): priority 1
                score = 0
                for segment in path.split('/'):
                    if segment.startswith(':'):
                        score += 2
                    elif segment == '*':
                        score += 1
                    else:
                        score += 3
                return score

        class RouterTrie:
            """Trie-based router for fast path matching"""

            def __init__(self):
                self.root = RouteNode()
                self.routes: Dict[str, Route] = {}

            def add_route(self, route: Route):
                """
                Add route to trie

                Args:
                    route: Route configuration

                Example:
                    /api/users/:id → ['api', 'users', ':id']
                """
                segments = [s for s in route.path.split('/') if s]
                node = self.root

                for segment in segments:
                    # Parameter segment (:id)
                    if segment.startswith(':'):
                        param_name = segment[1:]

                        # Find or create param node
                        param_node = None
                        for child_segment, child_node in node.children.items():
                            if child_node.is_param:
                                param_node = child_node
                                break

                        if param_node is None:
                            param_node = RouteNode()
                            param_node.is_param = True
                            param_node.param_name = param_name
                            node.children[f':{param_name}'] = param_node

                        node = param_node

                    # Wildcard segment (*)
                    elif segment == '*':
                        if '*' not in node.children:
                            wildcard_node = RouteNode()
                            wildcard_node.is_wildcard = True
                            node.children['*'] = wildcard_node
                        node = node.children['*']

                    # Static segment
                    else:
                        if segment not in node.children:
                            node.children[segment] = RouteNode()
                        node = node.children[segment]

                # Store route at leaf node
                node.route = route
                self.routes[route.path] = route

            def match_route(self, path: str, method: str) -> Tuple[Optional[Route], Dict[str, str]]:
                """
                Match request path to route

                Args:
                    path: Request path (e.g., /api/users/123)
                    method: HTTP method (GET, POST, etc.)

                Returns:
                    (matched_route, path_params)

                Example:
                    /api/users/123 → (route, {'id': '123'})
                """
                segments = [s for s in path.split('/') if s]
                return self._match_recursive(self.root, segments, 0, {}, method)

            def _match_recursive(
                self,
                node: RouteNode,
                segments: List[str],
                index: int,
                params: Dict[str, str],
                method: str
            ) -> Tuple[Optional[Route], Dict[str, str]]:
                """Recursive route matching with backtracking"""

                # Reached end of path
                if index == len(segments):
                    if node.route and method in node.route.methods:
                        return (node.route, params)
                    return (None, {})

                segment = segments[index]
                matches = []

                # Try static match (highest priority)
                if segment in node.children:
                    result = self._match_recursive(
                        node.children[segment],
                        segments,
                        index + 1,
                        params,
                        method
                    )
                    if result[0]:
                        matches.append((result[0].priority, result))

                # Try parameter match (medium priority)
                for child_segment, child_node in node.children.items():
                    if child_node.is_param:
                        new_params = params.copy()
                        new_params[child_node.param_name] = segment

                        result = self._match_recursive(
                            child_node,
                            segments,
                            index + 1,
                            new_params,
                            method
                        )
                        if result[0]:
                            matches.append((result[0].priority, result))

                # Try wildcard match (lowest priority)
                if '*' in node.children:
                    # Wildcard consumes all remaining segments
                    wildcard_node = node.children['*']
                    if wildcard_node.route and method in wildcard_node.route.methods:
                        matches.append((wildcard_node.route.priority, (wildcard_node.route, params)))

                # Return highest priority match
                if matches:
                    matches.sort(key=lambda x: x[0], reverse=True)
                    return matches[0][1]

                return (None, {})

            def remove_route(self, path: str):
                """Remove route from trie"""
                if path in self.routes:
                    del self.routes[path]
                    # Note: Not removing from trie to avoid complexity
                    # In production, periodically rebuild trie
        ```

        ---

        ## Connection Pooling

        **Problem:** Creating new TCP connections is expensive (3-way handshake = 3× RTT).

        **Solution:** Maintain pool of persistent connections to backend services.

        ```python
        import asyncio
        import aiohttp
        from typing import Dict

        class ConnectionPool:
            """HTTP connection pool for backend services"""

            def __init__(self, max_connections_per_host: int = 100):
                self.sessions: Dict[str, aiohttp.ClientSession] = {}
                self.max_connections = max_connections_per_host

            def get_session(self, host: str) -> aiohttp.ClientSession:
                """Get or create session for host"""
                if host not in self.sessions:
                    connector = aiohttp.TCPConnector(
                        limit_per_host=self.max_connections,
                        ttl_dns_cache=300,  # DNS cache
                        keepalive_timeout=60,  # Keep connections alive
                        enable_cleanup_closed=True
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=30,  # Total request timeout
                        connect=5,  # Connection timeout
                        sock_read=10  # Socket read timeout
                    )

                    self.sessions[host] = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout
                    )

                return self.sessions[host]

            async def forward_request(
                self,
                url: str,
                method: str,
                headers: dict,
                body: bytes
            ) -> aiohttp.ClientResponse:
                """
                Forward request to backend service

                Reuses existing connection if available
                """
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = f"{parsed.scheme}://{parsed.netloc}"

                session = self.get_session(host)

                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body
                ) as response:
                    return response

            async def close_all(self):
                """Close all sessions"""
                for session in self.sessions.values():
                    await session.close()
        ```

        ---

        ## Performance Optimizations

        | Optimization | Benefit | Implementation |
        |--------------|---------|----------------|
        | **Route trie** | O(m) matching vs O(n) | Prefix tree with priority |
        | **Connection pooling** | Reuse TCP connections | 100 connections per host |
        | **HTTP/2** | Multiplexing, header compression | Single connection, multiple streams |
        | **Response streaming** | Lower memory usage | Stream body chunks |
        | **Route compilation** | Faster regex matching | Pre-compile regex patterns |

    === "🔐 Authentication & Authorization"

        ## The Challenge

        **Problem:** Validate 500K JWT tokens/sec with <1ms latency. Asymmetric crypto (RS256) is expensive (0.5-1ms per verify).

        **Requirements:**

        - **Fast validation:** < 1ms p95
        - **Stateless:** No session database lookups
        - **Secure:** Asymmetric crypto (RS256/ES256)
        - **Cached:** Don't re-parse same token

        ---

        ## JWT Validation Flow

        ```python
        import jwt
        import hashlib
        import time
        from typing import Optional, Dict
        from functools import lru_cache

        class JWTAuthenticator:
            """JWT token validation with caching"""

            def __init__(self, redis_client, public_key: str):
                self.redis = redis_client
                self.public_key = public_key
                self.local_cache = {}  # In-memory cache
                self.cache_size = 10000  # LRU cache size

            async def authenticate(self, token: str) -> Optional[Dict]:
                """
                Authenticate JWT token with multi-level caching

                Args:
                    token: JWT token string

                Returns:
                    Parsed claims if valid, None otherwise

                Caching strategy:
                1. In-memory cache (< 0.1ms)
                2. Redis cache (< 1ms)
                3. Parse + verify signature (< 5ms)
                """
                # Level 1: In-memory cache (fastest)
                token_hash = self._hash_token(token)

                if token_hash in self.local_cache:
                    cached = self.local_cache[token_hash]
                    if cached['expires_at'] > time.time():
                        return cached['claims']
                    else:
                        del self.local_cache[token_hash]

                # Level 2: Redis cache
                cache_key = f"jwt:cache:{token_hash}"
                cached_claims = await self.redis.get(cache_key)

                if cached_claims:
                    claims = json.loads(cached_claims)

                    # Populate in-memory cache
                    self.local_cache[token_hash] = {
                        'claims': claims,
                        'expires_at': claims['exp']
                    }

                    # LRU eviction
                    if len(self.local_cache) > self.cache_size:
                        oldest = min(self.local_cache.items(), key=lambda x: x[1]['expires_at'])
                        del self.local_cache[oldest[0]]

                    return claims

                # Level 3: Parse and verify (slowest)
                try:
                    claims = jwt.decode(
                        token,
                        self.public_key,
                        algorithms=['RS256'],
                        options={
                            'verify_signature': True,
                            'verify_exp': True,
                            'verify_iat': True,
                            'require_exp': True,
                            'require_iat': True
                        }
                    )

                    # Cache in Redis
                    ttl = claims['exp'] - int(time.time())
                    if ttl > 0:
                        await self.redis.setex(
                            cache_key,
                            ttl,
                            json.dumps(claims)
                        )

                        # Cache in memory
                        self.local_cache[token_hash] = {
                            'claims': claims,
                            'expires_at': claims['exp']
                        }

                    return claims

                except jwt.ExpiredSignatureError:
                    logger.warning(f"Expired token: {token_hash}")
                    return None
                except jwt.InvalidTokenError as e:
                    logger.warning(f"Invalid token: {token_hash}, error: {e}")
                    return None

            def _hash_token(self, token: str) -> str:
                """Hash token for cache key"""
                return hashlib.sha256(token.encode()).hexdigest()[:16]
        ```

        ---

        ## Authorization (Scope-Based)

        ```python
        from typing import List, Set

        class AuthorizationService:
            """Check user permissions/scopes"""

            def __init__(self):
                # Route → required scopes mapping
                self.route_permissions = {
                    'GET:/api/users/:id': ['read:users'],
                    'POST:/api/users': ['write:users'],
                    'DELETE:/api/users/:id': ['delete:users', 'admin'],
                    'GET:/api/orders/:id': ['read:orders'],
                    'POST:/api/orders': ['write:orders'],
                }

            def authorize(
                self,
                route_key: str,
                user_scopes: List[str]
            ) -> bool:
                """
                Check if user has required scopes

                Args:
                    route_key: Route identifier (e.g., "GET:/api/users/123")
                    user_scopes: User's granted scopes from JWT

                Returns:
                    True if authorized, False otherwise
                """
                required_scopes = self.route_permissions.get(route_key, [])

                # No scopes required = public route
                if not required_scopes:
                    return True

                user_scope_set = set(user_scopes)

                # Check if user has any required scope (OR logic)
                for scope in required_scopes:
                    if scope in user_scope_set:
                        return True

                return False

            def authorize_all(
                self,
                route_key: str,
                user_scopes: List[str]
            ) -> bool:
                """
                Check if user has ALL required scopes (AND logic)
                """
                required_scopes = self.route_permissions.get(route_key, [])

                if not required_scopes:
                    return True

                user_scope_set = set(user_scopes)
                return all(scope in user_scope_set for scope in required_scopes)
        ```

        ---

        ## API Key Authentication

        **Alternative to JWT for server-to-server:**

        ```python
        import secrets
        import hashlib

        class APIKeyManager:
            """Manage API keys for service-to-service auth"""

            def __init__(self, db, redis):
                self.db = db
                self.redis = redis

            def generate_api_key(self, user_id: str, name: str) -> str:
                """
                Generate new API key

                Format: api_<prefix>_<random_bytes>
                Example: api_live_sk_1234567890abcdef
                """
                prefix = 'live' if self._is_production() else 'test'
                random_bytes = secrets.token_hex(24)
                api_key = f"api_{prefix}_sk_{random_bytes}"

                # Hash for storage (never store plaintext)
                key_hash = self._hash_key(api_key)

                # Store in database
                self.db.execute(
                    """INSERT INTO api_keys
                       (key_id, api_key_hash, user_id, name, created_at)
                       VALUES (%s, %s, %s, %s, NOW())""",
                    (secrets.token_hex(16), key_hash, user_id, name)
                )

                return api_key

            async def validate_api_key(self, api_key: str) -> Optional[Dict]:
                """
                Validate API key with caching

                Returns:
                    User info if valid, None otherwise
                """
                key_hash = self._hash_key(api_key)

                # Check Redis cache
                cache_key = f"apikey:cache:{key_hash}"
                cached = await self.redis.get(cache_key)

                if cached:
                    return json.loads(cached)

                # Database lookup
                result = self.db.query(
                    """SELECT k.user_id, k.scopes, k.rate_limit, k.revoked,
                              u.username, u.email
                       FROM api_keys k
                       JOIN users u ON k.user_id = u.user_id
                       WHERE k.api_key_hash = %s
                       AND (k.expires_at IS NULL OR k.expires_at > NOW())""",
                    (key_hash,)
                )

                if not result or result['revoked']:
                    return None

                user_info = {
                    'user_id': result['user_id'],
                    'username': result['username'],
                    'scopes': result['scopes'],
                    'rate_limit': result['rate_limit']
                }

                # Cache for 5 minutes
                await self.redis.setex(cache_key, 300, json.dumps(user_info))

                # Update last used timestamp (async)
                self.db.execute_async(
                    "UPDATE api_keys SET last_used_at = NOW() WHERE api_key_hash = %s",
                    (key_hash,)
                )

                return user_info

            def _hash_key(self, api_key: str) -> str:
                """Hash API key using SHA-256"""
                return hashlib.sha256(api_key.encode()).hexdigest()
        ```

        ---

        ## Performance Comparison

        | Auth Method | Latency | Pros | Cons |
        |-------------|---------|------|------|
        | **JWT (cached)** | < 0.1ms | Fast, stateless, no DB | Large token size (1-2 KB) |
        | **JWT (uncached)** | 1-5ms | Stateless, no DB | Crypto overhead, can't revoke |
        | **API Key (cached)** | < 1ms | Fast, revocable | Requires Redis lookup |
        | **API Key (uncached)** | 5-20ms | Revocable, simple | Database lookup required |
        | **OAuth 2.0** | 50-200ms | Standard, secure | Token introspection = extra call |

    === "🚦 Rate Limiting"

        ## The Challenge

        **Problem:** Limit requests per user across 100 distributed gateway nodes. 1M users × 1000 req/min = 16M req/sec at peak.

        **Requirements:**

        - **Distributed:** Shared counters across gateways
        - **Fast:** < 2ms overhead
        - **Accurate:** Don't allow bursts beyond limit
        - **Fair:** No single user can monopolize
        - **Graceful degradation:** If Redis down, allow requests

        ---

        ## Token Bucket Algorithm

        **Concept:**

        ```
        Bucket capacity: 1000 tokens
        Refill rate: 1000 tokens/minute = 16.67 tokens/second

        Request arrives:
        1. Refill tokens based on time elapsed
        2. If tokens >= 1:
           - Consume 1 token
           - Allow request
        3. Else:
           - Reject request (429)
        ```

        **Benefits:**

        - **Allows bursts:** Can use full capacity if bucket full
        - **Smooth over time:** Refills gradually
        - **Simple:** Single counter per user

        ---

        ## Implementation

        ```python
        import time
        import math
        from typing import Tuple

        class TokenBucketRateLimiter:
            """Token bucket rate limiter with Redis"""

            def __init__(self, redis_client):
                self.redis = redis_client

            async def is_allowed(
                self,
                key: str,
                capacity: int,
                refill_rate: float
            ) -> Tuple[bool, dict]:
                """
                Check if request is allowed

                Args:
                    key: Rate limit key (e.g., user_id, api_key)
                    capacity: Max tokens in bucket
                    refill_rate: Tokens per second

                Returns:
                    (allowed, metadata)

                Metadata:
                    - remaining: Tokens remaining
                    - reset_at: Timestamp when bucket refills to capacity
                    - retry_after: Seconds until next token available (if rejected)
                """
                now = time.time()

                # Lua script for atomic token bucket operations
                lua_script = """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])
                local requested = tonumber(ARGV[4])

                -- Get current state
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1])
                local last_refill = tonumber(bucket[2])

                -- Initialize if not exists
                if tokens == nil then
                    tokens = capacity
                    last_refill = now
                end

                -- Refill tokens based on elapsed time
                local elapsed = now - last_refill
                local refill_amount = elapsed * refill_rate
                tokens = math.min(capacity, tokens + refill_amount)

                -- Try to consume tokens
                local allowed = 0
                local remaining = tokens
                local retry_after = 0

                if tokens >= requested then
                    tokens = tokens - requested
                    allowed = 1
                else
                    retry_after = (requested - tokens) / refill_rate
                end

                remaining = tokens

                -- Update state
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity

                -- Return: allowed, remaining, retry_after
                return {allowed, remaining, retry_after}
                """

                # Execute Lua script
                result = await self.redis.eval(
                    lua_script,
                    1,  # Number of keys
                    f"ratelimit:token_bucket:{key}",
                    capacity,
                    refill_rate,
                    now,
                    1  # Tokens to consume per request
                )

                allowed = bool(result[0])
                remaining = int(result[1])
                retry_after = float(result[2])

                # Calculate reset time
                time_to_full = (capacity - remaining) / refill_rate
                reset_at = now + time_to_full

                metadata = {
                    'remaining': remaining,
                    'reset_at': int(reset_at),
                    'retry_after': int(math.ceil(retry_after)) if not allowed else 0
                }

                return allowed, metadata
        ```

        ---

        ## Sliding Window Log (More Accurate)

        **Problem:** Token bucket allows bursts. What if we want strict limits?

        **Solution:** Track every request timestamp in sliding window.

        ```python
        class SlidingWindowRateLimiter:
            """Sliding window log for accurate rate limiting"""

            def __init__(self, redis_client):
                self.redis = redis_client

            async def is_allowed(
                self,
                key: str,
                limit: int,
                window_seconds: int
            ) -> Tuple[bool, dict]:
                """
                Check if request is allowed using sliding window

                Args:
                    key: Rate limit key
                    limit: Max requests in window
                    window_seconds: Time window in seconds

                Example:
                    limit=1000, window=60 → 1000 requests per minute
                """
                now = time.time()
                window_start = now - window_seconds

                # Lua script for atomic sliding window operations
                lua_script = """
                local key = KEYS[1]
                local limit = tonumber(ARGV[1])
                local window_start = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])

                -- Remove old requests outside window
                redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

                -- Count requests in current window
                local count = redis.call('ZCARD', key)

                local allowed = 0
                if count < limit then
                    -- Add current request
                    redis.call('ZADD', key, now, now)
                    redis.call('EXPIRE', key, 3600)
                    allowed = 1
                    count = count + 1
                end

                -- Return: allowed, current_count
                return {allowed, count}
                """

                result = await self.redis.eval(
                    lua_script,
                    1,
                    f"ratelimit:sliding_window:{key}",
                    limit,
                    window_start,
                    now
                )

                allowed = bool(result[0])
                current_count = int(result[1])
                remaining = max(0, limit - current_count)

                metadata = {
                    'remaining': remaining,
                    'reset_at': int(now + window_seconds),
                    'retry_after': window_seconds if not allowed else 0
                }

                return allowed, metadata
        ```

        ---

        ## Hierarchical Rate Limiting

        **Multiple limits per user:**

        ```python
        class HierarchicalRateLimiter:
            """Multiple rate limits per user (per-second, per-minute, per-hour)"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.token_bucket = TokenBucketRateLimiter(redis_client)

            async def is_allowed(self, user_id: str, route: str) -> Tuple[bool, dict]:
                """
                Check multiple rate limits

                Rate limit tiers:
                1. Per-second: 100 req/sec (prevent burst attacks)
                2. Per-minute: 1000 req/min (normal usage)
                3. Per-hour: 10000 req/hour (daily quota)
                """
                limits = [
                    {'key': f'sec:{user_id}:{route}', 'capacity': 100, 'rate': 100},  # 100/sec
                    {'key': f'min:{user_id}:{route}', 'capacity': 1000, 'rate': 16.67},  # 1000/min
                    {'key': f'hour:{user_id}:{route}', 'capacity': 10000, 'rate': 2.78},  # 10K/hour
                ]

                # Check all limits (must pass all)
                for limit_config in limits:
                    allowed, metadata = await self.token_bucket.is_allowed(
                        limit_config['key'],
                        limit_config['capacity'],
                        limit_config['rate']
                    )

                    if not allowed:
                        return False, metadata

                # All limits passed
                return True, metadata
        ```

        ---

        ## Rate Limit Response Headers

        ```python
        def add_rate_limit_headers(response, metadata):
            """Add standard rate limit headers"""
            response.headers['X-RateLimit-Limit'] = str(metadata['limit'])
            response.headers['X-RateLimit-Remaining'] = str(metadata['remaining'])
            response.headers['X-RateLimit-Reset'] = str(metadata['reset_at'])

            if not metadata['allowed']:
                response.headers['Retry-After'] = str(metadata['retry_after'])

            return response
        ```

        ---

        ## Algorithm Comparison

        | Algorithm | Accuracy | Memory | Pros | Cons |
        |-----------|----------|--------|------|------|
        | **Token Bucket** | Good | Low | Simple, allows bursts | Not strictly accurate |
        | **Sliding Window Log** | Excellent | High | Precise, no bursts | O(n) memory per user |
        | **Fixed Window** | Poor | Low | Very simple | Burst at window boundary |
        | **Sliding Window Counter** | Good | Low | Accurate + efficient | Complex implementation |

    === "⚡ Circuit Breaker"

        ## The Challenge

        **Problem:** Backend service fails (503 errors). Gateway keeps forwarding requests, amplifying failure. **Cascading failure** brings down entire system.

        **Requirements:**

        - **Fast failure detection:** Open circuit within seconds
        - **Automatic recovery:** Close circuit when service healthy
        - **Graceful degradation:** Return cached/default response
        - **Adaptive thresholds:** Different services have different error rates

        ---

        ## Circuit Breaker States

        ```python
        from enum import Enum
        import time
        import asyncio

        class CircuitState(Enum):
            CLOSED = "closed"      # Normal operation
            OPEN = "open"          # Circuit tripped, fail fast
            HALF_OPEN = "half_open"  # Testing recovery

        class CircuitBreaker:
            """Adaptive circuit breaker with Redis state"""

            def __init__(
                self,
                service_name: str,
                redis_client,
                failure_threshold: float = 0.5,  # 50% error rate
                success_threshold: float = 0.8,  # 80% success rate to close
                timeout_seconds: int = 30,
                window_size: int = 100,  # Sample size
                half_open_max_calls: int = 10  # Test requests in half-open
            ):
                self.service_name = service_name
                self.redis = redis_client
                self.failure_threshold = failure_threshold
                self.success_threshold = success_threshold
                self.timeout_seconds = timeout_seconds
                self.window_size = window_size
                self.half_open_max_calls = half_open_max_calls

                # Redis keys
                self.state_key = f"circuit:{service_name}:state"
                self.failures_key = f"circuit:{service_name}:failures"
                self.successes_key = f"circuit:{service_name}:successes"
                self.opened_at_key = f"circuit:{service_name}:opened_at"
                self.half_open_calls_key = f"circuit:{service_name}:half_open_calls"

            async def call(self, func, *args, **kwargs):
                """
                Execute function with circuit breaker protection

                Args:
                    func: Async function to call (e.g., backend request)

                Returns:
                    Function result or raises CircuitOpenError
                """
                state = await self._get_state()

                # Circuit OPEN - fail fast
                if state == CircuitState.OPEN:
                    # Check if timeout elapsed
                    opened_at = await self.redis.get(self.opened_at_key)
                    if opened_at and time.time() - float(opened_at) >= self.timeout_seconds:
                        # Transition to HALF_OPEN
                        await self._set_state(CircuitState.HALF_OPEN)
                        await self.redis.set(self.half_open_calls_key, 0)
                    else:
                        raise CircuitOpenError(f"Circuit breaker open for {self.service_name}")

                # Circuit HALF_OPEN - limited calls
                if state == CircuitState.HALF_OPEN:
                    calls = await self.redis.incr(self.half_open_calls_key)
                    if calls > self.half_open_max_calls:
                        raise CircuitOpenError(f"Circuit breaker half-open, max calls exceeded")

                # Execute function
                try:
                    result = await func(*args, **kwargs)
                    await self._record_success()
                    return result

                except Exception as e:
                    await self._record_failure()
                    raise e

            async def _record_success(self):
                """Record successful call"""
                # Add to sliding window
                await self.redis.lpush(self.successes_key, time.time())
                await self.redis.ltrim(self.successes_key, 0, self.window_size - 1)
                await self.redis.expire(self.successes_key, 60)

                state = await self._get_state()

                # HALF_OPEN → CLOSED if enough successes
                if state == CircuitState.HALF_OPEN:
                    success_count = await self.redis.llen(self.successes_key)
                    total = success_count + await self.redis.llen(self.failures_key)

                    if total >= 10:  # Minimum sample size
                        success_rate = success_count / total
                        if success_rate >= self.success_threshold:
                            await self._close_circuit()

            async def _record_failure(self):
                """Record failed call"""
                # Add to sliding window
                await self.redis.lpush(self.failures_key, time.time())
                await self.redis.ltrim(self.failures_key, 0, self.window_size - 1)
                await self.redis.expire(self.failures_key, 60)

                state = await self._get_state()

                # CLOSED → OPEN if too many failures
                if state == CircuitState.CLOSED:
                    failure_count = await self.redis.llen(self.failures_key)
                    success_count = await self.redis.llen(self.successes_key)
                    total = failure_count + success_count

                    if total >= 20:  # Minimum sample size
                        failure_rate = failure_count / total
                        if failure_rate >= self.failure_threshold:
                            await self._open_circuit()

                # HALF_OPEN → OPEN if any failure
                elif state == CircuitState.HALF_OPEN:
                    await self._open_circuit()

            async def _get_state(self) -> CircuitState:
                """Get current circuit state"""
                state = await self.redis.get(self.state_key)
                return CircuitState(state) if state else CircuitState.CLOSED

            async def _set_state(self, state: CircuitState):
                """Set circuit state"""
                await self.redis.set(self.state_key, state.value)

            async def _open_circuit(self):
                """Open circuit (fail fast)"""
                await self._set_state(CircuitState.OPEN)
                await self.redis.set(self.opened_at_key, time.time())
                logger.warning(f"Circuit breaker OPENED for {self.service_name}")

            async def _close_circuit(self):
                """Close circuit (normal operation)"""
                await self._set_state(CircuitState.CLOSED)
                await self.redis.delete(self.opened_at_key)
                await self.redis.delete(self.half_open_calls_key)
                await self.redis.delete(self.failures_key)
                await self.redis.delete(self.successes_key)
                logger.info(f"Circuit breaker CLOSED for {self.service_name}")

        class CircuitOpenError(Exception):
            """Raised when circuit breaker is open"""
            pass
        ```

        ---

        ## Fallback Strategies

        **When circuit is open, what to return?**

        ```python
        class FallbackHandler:
            """Fallback responses when circuit breaker is open"""

            def __init__(self, cache):
                self.cache = cache

            async def get_fallback_response(
                self,
                service: str,
                method: str,
                path: str
            ) -> Optional[dict]:
                """
                Get fallback response when service unavailable

                Strategies:
                1. Cached response (stale data OK)
                2. Default response
                3. 503 error with retry-after
                """
                # Strategy 1: Return cached response (even if expired)
                cache_key = f"cache:{method}:{path}"
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.info(f"Returning stale cache for {path}")
                    return {
                        'data': json.loads(cached),
                        'headers': {
                            'X-Cache': 'HIT-STALE',
                            'X-Circuit-State': 'open'
                        }
                    }

                # Strategy 2: Default response (for non-critical endpoints)
                if service == 'recommendation-service':
                    logger.info(f"Returning default recommendations")
                    return {
                        'data': {'recommendations': []},
                        'headers': {
                            'X-Circuit-State': 'open',
                            'X-Fallback': 'default'
                        }
                    }

                # Strategy 3: Fail with 503 (for critical endpoints)
                return None
        ```

        ---

        ## Integration with Gateway

        ```python
        class GatewayRequestHandler:
            """Main request handler with circuit breaker"""

            def __init__(self, router, circuit_breakers, fallback_handler):
                self.router = router
                self.breakers = circuit_breakers
                self.fallback = fallback_handler

            async def handle_request(self, request):
                """Handle incoming request with circuit breaker"""
                # Route matching
                route, params = self.router.match_route(request.path, request.method)
                if not route:
                    return Response(404, "Not found")

                service = route.service
                circuit_breaker = self.breakers.get(service)

                # Execute with circuit breaker
                try:
                    response = await circuit_breaker.call(
                        self._forward_request,
                        service,
                        request
                    )
                    return response

                except CircuitOpenError:
                    # Try fallback
                    fallback_response = await self.fallback.get_fallback_response(
                        service,
                        request.method,
                        request.path
                    )

                    if fallback_response:
                        return Response(
                            200,
                            fallback_response['data'],
                            headers=fallback_response['headers']
                        )
                    else:
                        return Response(
                            503,
                            "Service unavailable. Circuit breaker open.",
                            headers={'Retry-After': '30'}
                        )

            async def _forward_request(self, service, request):
                """Forward request to backend service"""
                # Service discovery, load balancing, etc.
                instance = await self._get_service_instance(service)
                response = await self._http_client.request(
                    method=request.method,
                    url=f"http://{instance}/{request.path}",
                    headers=request.headers,
                    body=request.body
                )
                return response
        ```

        ---

        ## Circuit Breaker Benefits

        | Benefit | Impact |
        |---------|--------|
        | **Prevent cascading failures** | Failed service doesn't bring down gateway |
        | **Fast failure** | Don't wait for timeouts (30s → 1ms) |
        | **Automatic recovery** | Self-healing without manual intervention |
        | **Resource protection** | Free up connections/threads |
        | **Better UX** | Return fallback data vs. hanging request |

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling API Gateway from 10K to 500K requests/sec.

    **Scaling challenges at 500K req/sec:**

    - **Latency:** < 10ms overhead (routing + auth + rate limit)
    - **Throughput:** 100 gateway nodes × 5K req/sec each
    - **State management:** Redis cluster for rate limiting
    - **Configuration sync:** 50K routes across all nodes
    - **SSL termination:** TLS handshake overhead

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **JWT validation** | ✅ Yes | Multi-level caching (99% cache hit), asymmetric crypto |
    | **Rate limiting** | ✅ Yes | Redis cluster (100K ops/sec per node), Lua scripts (atomic) |
    | **Route matching** | 🟢 No | Trie data structure (O(m) lookup), compiled in memory |
    | **SSL termination** | ✅ Yes | Offload to load balancer, TLS session resumption |
    | **Backend connections** | ✅ Yes | Connection pooling (100 per host), HTTP/2 multiplexing |
    | **Redis latency** | 🟡 Moderate | Co-located Redis (< 1ms), pipelining, local cache |

    ---

    ## Performance Optimizations

    ### 1. Request Path Optimization

    **Goal:** Reduce latency from 50ms to <10ms.

    | Optimization | Before | After | Savings |
    |--------------|--------|-------|---------|
    | JWT caching | 5ms | 0.1ms | 4.9ms |
    | Rate limit (Redis) | 3ms | 1ms | 2ms |
    | Route matching (linear) | 2ms | 0.1ms | 1.9ms |
    | Backend connection setup | 10ms | 0.5ms | 9.5ms |
    | Response buffering | 5ms | 1ms | 4ms |
    | **Total** | **25ms** | **2.7ms** | **22.3ms** |

    ---

    ### 2. Memory Optimization

    **Per-node memory usage:**

    ```python
    # Before optimization: 12 GB per node
    - Route trie: 500 MB (50K routes × 10 KB)
    - JWT cache: 5 GB (500K tokens × 10 KB)
    - Response cache: 4 GB (400K responses × 10 KB)
    - Connection pools: 2 GB (10K connections × 200 KB)
    - OS/runtime: 500 MB

    # After optimization: 5 GB per node
    - Route trie: 100 MB (compact encoding)
    - JWT cache: 2 GB (cache claims only, not full token)
    - Response cache: 1 GB (cache headers only, stream body)
    - Connection pools: 1 GB (smaller buffers)
    - OS/runtime: 1 GB (JVM tuning)
    ```

    **Optimization techniques:**

    ```python
    class CompactJWTCache:
        """Cache only parsed claims, not full token"""

        def cache_token(self, token: str, claims: dict):
            # Before: Store full token (1-2 KB)
            # self.cache[token] = {'token': token, 'claims': claims}

            # After: Store hash + claims only (200 bytes)
            token_hash = hashlib.sha256(token.encode()).digest()[:8]
            self.cache[token_hash] = {
                'sub': claims['sub'],  # user_id
                'exp': claims['exp'],  # expiry
                'scopes': claims.get('scope', '').split()
            }
    ```

    ---

    ### 3. CPU Optimization

    **CPU-intensive operations:**

    | Operation | CPU Time | Optimization |
    |-----------|----------|--------------|
    | JWT signature verification | 40% | Cache validated tokens (99% hit rate) |
    | Route regex matching | 15% | Pre-compile regex, use trie for path matching |
    | JSON parsing | 20% | Lazy parsing (parse on demand), binary protocols |
    | Logging | 10% | Async logging, sample logs (1% in production) |
    | SSL/TLS | 15% | Offload to LB, use TLS session resumption |

    ---

    ### 4. Redis Optimization

    **Reduce Redis latency from 3ms to 1ms:**

    ```python
    class OptimizedRedisClient:
        """Optimized Redis access patterns"""

        def __init__(self, redis_cluster):
            self.redis = redis_cluster
            self.local_cache = {}  # L1 cache

        async def get_with_local_cache(self, key: str):
            """Two-level cache: local memory → Redis"""
            # L1: Local cache (< 0.1ms)
            if key in self.local_cache:
                cached = self.local_cache[key]
                if cached['expires_at'] > time.time():
                    return cached['value']

            # L2: Redis (< 1ms)
            value = await self.redis.get(key)
            if value:
                self.local_cache[key] = {
                    'value': value,
                    'expires_at': time.time() + 10  # 10s local cache
                }

            return value

        async def batch_get(self, keys: List[str]):
            """Use MGET for batch operations"""
            # Before: N individual GETs = N × 1ms = 10ms for 10 keys
            # results = [await self.redis.get(k) for k in keys]

            # After: Single MGET = 1ms for 10 keys
            results = await self.redis.mget(keys)
            return results

        async def pipeline_commands(self, commands: List[Tuple]):
            """Use pipeline for multiple operations"""
            # Before: 5 commands × 1ms = 5ms
            # After: 1 pipelined batch = 1ms
            pipeline = self.redis.pipeline()
            for cmd, args in commands:
                getattr(pipeline, cmd)(*args)
            results = await pipeline.execute()
            return results
    ```

    ---

    ## Horizontal Scaling

    **Scaling from 10 to 100 gateway nodes:**

    ```
    Traffic distribution:
    500K req/sec ÷ 100 nodes = 5K req/sec per node

    Node specifications:
    - CPU: 8 cores (50% utilization)
    - Memory: 8 GB (5 GB used, 3 GB free)
    - Network: 10 Gbps (2 Gbps used)
    - Connections: 10K concurrent

    Load balancer:
    - Algorithm: Least connections
    - Health checks: HTTP /health every 5s
    - Unhealthy threshold: 3 failures
    - Auto-scaling: CPU > 70% → add node
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 500K req/sec:**

    | Component | Instances | Unit Cost | Monthly Cost |
    |-----------|-----------|-----------|--------------|
    | **Gateway nodes** | 100 × c5.2xlarge | $0.34/hr | $24,480 |
    | **Redis cluster** | 20 nodes (sharded) | $0.50/hr | $7,200 |
    | **Load balancer** | 2 × ALB | $30/month | $60 |
    | **Config DB** | 1 × db.t3.medium | $0.068/hr | $49 |
    | **Monitoring** | Prometheus + Grafana | $500/month | $500 |
    | **Logging** | ELK stack (3 nodes) | $1000/month | $1,000 |
    | **Total** | | | **$33,289/month** |

    **Cost per million requests:**

    ```
    Monthly requests: 500K/sec × 86,400 sec/day × 30 days = 1.3 trillion
    Cost: $33,289 / 1,300,000 = $0.026 per million requests
    ```

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Gateway Latency (P95)** | < 10ms | > 20ms |
    | **Backend Latency (P95)** | < 100ms | > 500ms |
    | **Error Rate** | < 0.1% | > 1% |
    | **Rate Limit Hit Rate** | < 5% | > 10% |
    | **Circuit Breaker Open** | 0 | Any service open > 5 min |
    | **Cache Hit Rate** | > 90% | < 80% |
    | **CPU Utilization** | < 70% | > 85% |
    | **Redis Latency** | < 2ms | > 5ms |

    ---

    ## Health Checks

    ```python
    from fastapi import FastAPI, Response
    import time

    app = FastAPI()

    @app.get("/health")
    async def health_check():
        """
        Load balancer health check endpoint

        Checks:
        1. Redis connectivity
        2. Config service reachable
        3. Memory usage < 90%
        4. CPU usage < 95%
        """
        checks = {
            'redis': await check_redis(),
            'config_service': await check_config_service(),
            'memory': check_memory(),
            'cpu': check_cpu()
        }

        all_healthy = all(checks.values())

        return Response(
            content=json.dumps({
                'status': 'healthy' if all_healthy else 'unhealthy',
                'checks': checks,
                'timestamp': time.time()
            }),
            status_code=200 if all_healthy else 503,
            media_type='application/json'
        )

    @app.get("/ready")
    async def readiness_check():
        """
        Kubernetes readiness probe

        Returns 200 when gateway is ready to accept traffic
        """
        # Check if config loaded
        if not router.routes:
            return Response(status_code=503)

        # Check if can reach Redis
        if not await check_redis():
            return Response(status_code=503)

        return Response(status_code=200)
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Trie-based routing:** O(m) path matching, handles 50K routes efficiently
    2. **Multi-level caching:** JWT tokens cached in memory + Redis (99% hit rate)
    3. **Token bucket rate limiting:** Distributed counters in Redis with Lua scripts
    4. **Adaptive circuit breaker:** Prevents cascading failures, auto-recovery
    5. **Connection pooling:** Reuse HTTP connections, HTTP/2 multiplexing
    6. **Stateless design:** No session state, JWT for authentication
    7. **Configuration-driven:** Dynamic routing without code deploys

    ---

    ## Interview Tips

    ✅ **Start with functional requirements** - Routing, auth, rate limiting are must-haves

    ✅ **Discuss latency budget** - Gateway adds <10ms, critical for UX

    ✅ **Rate limiting algorithms** - Token bucket vs sliding window trade-offs

    ✅ **Circuit breaker states** - CLOSED → OPEN → HALF_OPEN state machine

    ✅ **Caching strategy** - Multi-level cache (memory → Redis → origin)

    ✅ **Security considerations** - JWT validation, scope-based auth, API keys

    ✅ **Scalability** - Horizontal scaling, Redis cluster, load balancing

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle rate limiting across multiple gateways?"** | Redis counters with Lua scripts (atomic), token bucket algorithm, eventual consistency OK |
    | **"What if Redis goes down?"** | Degrade gracefully (allow requests), local rate limiting, circuit breaker for Redis |
    | **"How to implement circuit breaker?"** | Track error rate in sliding window, open circuit at 50% errors, half-open testing after timeout |
    | **"How to validate JWT tokens efficiently?"** | Cache parsed tokens (5 min TTL), cache public keys, asymmetric crypto (RS256) |
    | **"How to route to multiple backend versions?"** | Weighted routing (90% v1, 10% v2), header-based routing, canary deployments |
    | **"How to aggregate multiple backend calls?"** | Parallel requests with async/await, timeout per request, return partial data if one fails |
    | **"How to transform requests/responses?"** | Plugin architecture, Lua scripts (Kong), middleware chain pattern |

    ---

    ## Real-World Examples

    | Company | Gateway Solution | Scale |
    |---------|-----------------|-------|
    | **Netflix** | Zuul | 1M+ req/sec, 1000+ microservices |
    | **Kong** | Kong Gateway | 100K+ req/sec per node |
    | **AWS** | API Gateway | Multi-million req/sec, serverless |
    | **Cloudflare** | Cloudflare Workers | 25M+ req/sec global |
    | **Google** | Apigee | Enterprise API management |

    ---

    ## Further Optimizations

    **Beyond scope but worth mentioning:**

    - **GraphQL Gateway:** Combine multiple REST APIs into GraphQL schema
    - **gRPC Support:** Binary protocol for internal microservices
    - **WebSocket Proxy:** Real-time bidirectional communication
    - **Request Deduplication:** Cache identical concurrent requests
    - **Adaptive Rate Limiting:** ML-based anomaly detection
    - **Geographic Routing:** Route to nearest data center
    - **A/B Testing:** Route percentage of traffic to experimental backends

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Kong, AWS, Google (Apigee), Netflix, Cloudflare

---

*Master this problem and you'll be ready for: Service Mesh (Istio), Load Balancer, CDN, Reverse Proxy designs*
