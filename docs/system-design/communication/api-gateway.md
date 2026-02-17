# API Gateway

Every request from the outside world must somehow find its way to the correct
microservice. Without an API gateway, clients must know the address of every service,
handle authentication themselves, and cope with different protocols and data formats
across services. An API gateway is a single entry point that sits between clients and
your backend services, routing requests, enforcing policies, and shielding internal
architecture from the outside world.

Think of it as the front desk of a large hotel. Guests (clients) don't wander the
hallways looking for housekeeping, room service, or the concierge. They go to the front
desk, which routes their request to the right department. The hotel can reorganize
departments, add new ones, or merge existing ones without guests noticing — because the
front desk abstracts it all away.

---

## Why You Need an API Gateway

```
WITHOUT GATEWAY: Clients coupled to every service
┌────────┐──→ User Service (auth, rate limit, transform)
│ Mobile  │──→ Order Service (auth, rate limit, transform)
│ App     │──→ Product Service (auth, rate limit, transform)
└────────┘──→ Payment Service (auth, rate limit, transform)
              Every service duplicates cross-cutting concerns.
              Client must know every service address.

WITH GATEWAY: Single entry point
┌────────┐      ┌─────────────┐──→ User Service
│ Mobile  │─────→│ API Gateway  │──→ Order Service
│ App     │     │ auth, rate   │──→ Product Service
└────────┘     │ limit, route │──→ Payment Service
               └─────────────┘
               Cross-cutting concerns handled once.
               Client knows one address.
```

The gateway centralizes cross-cutting concerns that would otherwise be duplicated
across every service: authentication, rate limiting, request logging, protocol
translation, and response caching.

---

=== "Core Responsibilities"

    ## What the Gateway Does

    ### Request Routing

    The most fundamental job: map incoming requests to the correct backend service. This
    can be path-based, header-based, or method-based.

    ```
    Routing Rules:
    /api/users/*    → User Service     (port 8001)
    /api/orders/*   → Order Service    (port 8002)
    /api/products/* → Product Service  (port 8003)
    /api/v2/*       → New API cluster  (canary routing)

    Headers-based:
    X-Client: mobile    → Mobile-optimized service
    X-Client: web       → Standard service
    ```

    ### Authentication & Authorization

    The gateway validates identity (who are you?) before requests ever reach backend
    services. This means services can trust that any request they receive has already
    been authenticated.

    ```
    Client ──→ Gateway ──→ Backend
               │
               ├─ Validate JWT token
               ├─ Check token expiration
               ├─ Verify token signature
               ├─ Extract user claims
               └─ Add X-User-ID header to downstream request

    Backend receives:
      Original request + X-User-ID: 12345
      No need to validate token again
    ```

    ### Rate Limiting & Throttling

    Protect backend services from being overwhelmed by limiting how many requests a
    client can make in a given time window.

    | Algorithm | How It Works | Best For |
    |---|---|---|
    | **Fixed window** | Count requests per fixed time interval (e.g., 100/minute) | Simple APIs |
    | **Sliding window** | Rolling count over past N seconds | Smoother limiting |
    | **Token bucket** | Tokens added at steady rate, each request costs a token | Burst-tolerant APIs |
    | **Leaky bucket** | Requests queued, processed at fixed rate | Steady throughput |

    **Stripe** allows 100 requests per second per API key. **Twitter** allows 300 tweets
    per 3 hours per user. **GitHub** allows 5,000 requests per hour per authenticated
    user. All enforced at the gateway layer.

    ### Request/Response Transformation

    The gateway can modify requests and responses in flight — translating protocols,
    aggregating responses from multiple services, or stripping internal fields.

    ```
    Client requests: GET /api/user-dashboard

    Gateway fans out:
      → User Service:    GET /users/123         → { name, email }
      → Order Service:   GET /orders?user=123   → { recent_orders }
      → Loyalty Service: GET /points?user=123   → { points, tier }

    Gateway aggregates and returns:
    {
      "user": { "name": "...", "email": "..." },
      "recent_orders": [...],
      "loyalty": { "points": 2500, "tier": "gold" }
    }

    Client makes ONE call instead of THREE.
    ```

    ### Other Cross-Cutting Concerns

    - **Caching** — cache frequent read responses to reduce backend load
    - **Circuit breaking** — stop sending traffic to failing services
    - **Request logging** — centralized access logs for all API traffic
    - **CORS handling** — manage cross-origin policies in one place
    - **SSL termination** — handle TLS at the gateway, plain HTTP internally
    - **Compression** — gzip responses before sending to clients

=== "Gateway Patterns"

    ## Common Gateway Patterns

    ### Simple Reverse Proxy

    The simplest pattern: the gateway routes requests to services without modification.
    NGINX and HAProxy have served this role for decades.

    ```
    Client ──→ NGINX ──→ Service A
                    ├──→ Service B
                    └──→ Service C
    ```

    Good for: small teams, simple routing, when you mainly need load balancing and
    SSL termination.

    ### Backend for Frontend (BFF)

    Different clients need different APIs. A mobile app needs compact payloads and
    minimal round trips. A web app can handle richer responses. An internal admin
    dashboard needs detailed data.

    Instead of one gateway serving all clients, create a **separate gateway per client
    type**, each tailored to that client's needs.

    ```
    ┌──────────┐    ┌──────────────┐
    │ Mobile   │───→│ Mobile BFF   │──→ Services
    │ App      │    └──────────────┘
    └──────────┘
    ┌──────────┐    ┌──────────────┐
    │ Web      │───→│  Web BFF     │──→ Services
    │ App      │    └──────────────┘
    └──────────┘
    ┌──────────┐    ┌──────────────┐
    │ Partner  │───→│ Partner BFF  │──→ Services
    │ API      │    └──────────────┘
    └──────────┘
    ```

    **SoundCloud** pioneered the BFF pattern. Their mobile app needed a fundamentally
    different API shape than their web app, and trying to serve both from a single
    gateway created complexity and performance problems. Separate BFFs let each team
    optimize independently.

    **Netflix** has separate gateways for their TV app, mobile app, and web player. Each
    gateway aggregates different combinations of backend services and optimizes response
    payloads for the specific client's capabilities and screen size.

    ### Gateway Aggregation

    The gateway composes responses from multiple backend services into a single response.
    This reduces client round trips and simplifies client code.

    ```
    Single client request: GET /api/product/456

    Gateway calls in parallel:
      → Product Service:  product details
      → Review Service:   reviews for product 456
      → Inventory Service: stock level
      → Pricing Service:   current price with discounts

    Gateway returns combined response in ~max(latencies)
    instead of client making 4 sequential calls = sum(latencies)
    ```

    The risk: the gateway becomes a "God gateway" that contains business logic. Keep
    aggregation to simple data composition — if you're writing if/else logic in the
    gateway, that logic belongs in a service.

    ### Gateway Offloading

    Move resource-intensive work from services to the gateway: SSL termination, response
    compression, static content serving, request validation. This lets backend services
    focus on business logic and keeps their resource usage predictable.

=== "Tools Comparison"

    ## Popular API Gateway Tools

    | Tool | Type | Strengths | Weaknesses | Used By |
    |---|---|---|---|---|
    | **NGINX** | Reverse proxy / gateway | Extremely fast, battle-tested, huge ecosystem | Configuration-based, limited dynamic routing | Dropbox, Netflix, WordPress |
    | **Kong** | Full API gateway | Plugin ecosystem, Lua extensibility, admin API | Resource-heavy, complex clustering | Nasdaq, Honeywell |
    | **AWS API Gateway** | Managed service | Zero ops, auto-scaling, IAM integration | Vendor lock-in, 29-second timeout, cost at scale | Serverless architectures |
    | **Envoy** | Service proxy | L4/L7, gRPC-native, observability built-in | Complex configuration, steep learning curve | Lyft, Airbnb, Salesforce |
    | **Traefik** | Cloud-native gateway | Auto-discovery, Docker/K8s native, Let's Encrypt | Less mature plugin ecosystem | Kubernetes-native teams |
    | **Azure API Management** | Managed service | Developer portal, policy engine | Vendor lock-in, complex pricing | Azure-centric organizations |

    ### How to Choose

    ```
    Are you on a single cloud?
    ├── AWS + serverless → AWS API Gateway
    ├── Azure → Azure API Management
    └── Multi-cloud or on-prem?
        │
        ├── Need plugin ecosystem + admin UI?
        │   └── Kong
        ├── Need gRPC + service mesh integration?
        │   └── Envoy (often with Istio)
        ├── Kubernetes-native with auto-discovery?
        │   └── Traefik
        └── Simple routing + maximum performance?
            └── NGINX
    ```

=== "Performance & Anti-Patterns"

    ## Performance Considerations

    The gateway is on the critical path of every request. Any latency it adds is felt by
    every user.

    **Latency budget:** A well-configured NGINX or Envoy gateway adds 1-5ms of latency.
    A gateway doing JWT validation, rate limiting, and logging typically adds 5-15ms.
    Response aggregation from multiple services adds the latency of the slowest backend
    call.

    **Connection pooling:** The gateway should maintain persistent connections to backend
    services. Establishing a new TCP connection costs 1-3ms; TLS adds another 5-20ms.
    With connection pooling, requests reuse existing connections.

    **Caching:** Cache responses for idempotent GET requests. Even a 30-second cache on
    frequently accessed endpoints can reduce backend load by 80-90%. Use cache headers
    (ETag, Cache-Control) to control freshness.

    ```
    Without connection pooling:
    Client → Gateway → [TCP handshake 3ms] → [TLS 15ms] → Service
    Total overhead: ~18ms per request

    With connection pooling:
    Client → Gateway → [reuse connection 0ms] → Service
    Total overhead: ~1ms per request
    ```

    ## Anti-Patterns

    ### The God Gateway

    **Problem:** Business logic creeps into the gateway — data validation, workflow
    orchestration, complex transformations. The gateway becomes a monolith that every
    team depends on.

    **Symptom:** Gateway deployments require coordination across teams. Changes to one
    service's routing break another service.

    **Fix:** The gateway should only handle cross-cutting concerns (auth, rate limiting,
    routing). Any logic specific to a business domain belongs in a service.

    ### No Fallback for Failures

    **Problem:** If a backend service is down, the gateway returns a 500 error to the
    client, even when partial data could be served.

    **Fix:** Implement circuit breakers and fallback responses. If the review service is
    down, return the product details without reviews rather than failing the entire
    request. Netflix's gateway returns cached or default data for non-critical services.

    ### Tight Coupling to Service Internals

    **Problem:** The gateway knows about internal service implementation details —
    specific database fields, internal error formats, or service-to-service protocols.

    **Fix:** Services should expose well-defined API contracts. The gateway routes based
    on URL paths and headers, not internal implementation details.

    ### Single Gateway for Everything

    **Problem:** One gateway serves mobile, web, partners, and internal services. Each
    client type has different needs, leading to complex conditional routing.

    **Fix:** Use the BFF pattern — one gateway per client type. Or use gateway tiers:
    an external gateway for public traffic and an internal gateway for service-to-service
    communication.

---

## Real-World Architecture: Netflix Zuul

Netflix processes billions of API requests per day through their gateway, Zuul. Their
architecture evolved through multiple generations:

**Zuul 1** (2013): Servlet-based, blocking I/O. Handled Netflix's scale for years but
connection-per-thread model limited throughput.

**Zuul 2** (2018): Rewritten with Netty for non-blocking I/O. Handles 100,000+
concurrent connections per instance. Key capabilities:

- **Request passport:** Every request gets a unique ID tracked through all services
- **Canary routing:** Routes 1-5% of traffic to new deployments for validation
- **Adaptive retries:** Automatically retries failed requests to different instances
- **Load shedding:** Drops low-priority requests under extreme load to protect critical paths

Netflix's gateway handles authentication, routing, canary testing, and observability
but deliberately avoids business logic — that stays in the backend services.

---

## Key Takeaways

1. **An API gateway is essential for microservices** once you have more than 3-4
   services. Without it, cross-cutting concerns are duplicated across every service and
   clients are coupled to internal architecture.

2. **Keep the gateway thin.** Its job is routing and cross-cutting concerns, not
   business logic. If you're writing domain-specific code in the gateway, that code
   belongs in a service.

3. **Use BFF pattern when clients diverge.** Mobile, web, and partner APIs have
   fundamentally different needs. Separate gateways let each client team move
   independently.

4. **The gateway is a single point of failure** — design accordingly. Run multiple
   instances behind a load balancer, implement health checks, and have a fallback plan
   (even if it's just returning cached responses).

5. **Measure gateway latency obsessively.** Every millisecond the gateway adds is felt
   by every user on every request. Profile regularly and optimize the hot path.

6. **Don't build your own** unless you have Netflix-scale problems. NGINX, Kong, or
   your cloud provider's managed gateway handles 99% of use cases with far less
   operational burden than a custom solution.

---

## Related Topics

- [Load Balancers](../networking/load-balancers.md) — distributing traffic across gateway instances
- [Authentication](../security/authentication.md) — identity verification at the gateway
- [Rate Limiting](../security/api-security.md) — throttling strategies enforced by the gateway
- [Microservices](../architecture/microservices.md) — the architecture pattern that necessitates gateways
- [Messaging Patterns](messaging/patterns.md) — async communication as an alternative to gateway aggregation
