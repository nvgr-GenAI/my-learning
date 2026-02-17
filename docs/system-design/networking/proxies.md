# Proxies

A proxy is a server that sits between two parties in a network conversation, intercepting and potentially modifying requests and responses as they pass through. The word comes from "procuracy" — acting on behalf of someone else. In networking, proxies act on behalf of either the client (forward proxy) or the server (reverse proxy), and this distinction matters enormously.

Proxies are everywhere in modern infrastructure — often invisibly. Every time you visit a website through Cloudflare, a reverse proxy handles your request. Every time your company filters your internet access, a forward proxy is at work. Understanding proxy patterns is fundamental to system design because they solve problems that neither clients nor servers can solve alone.

---

## Forward Proxy vs Reverse Proxy

The distinction is simple but critical: **who does the proxy represent?**

```
Forward Proxy (represents the client):

Client ──→ Forward Proxy ──→ Internet ──→ Server
              │
   "I'll make the request          Server sees the proxy's IP,
    on your behalf"                 not the client's


Reverse Proxy (represents the server):

Client ──→ Internet ──→ Reverse Proxy ──→ Backend Servers
                              │
                   "I'll handle the request    Client sees the proxy's IP,
                    for our servers"            not the backend's
```

A **forward proxy** is configured by the client (or the client's network). The client knows it's using a proxy and sends requests to it intentionally. The server on the other end has no idea a proxy is involved — it just sees a request from the proxy's IP address.

A **reverse proxy** is configured by the server operator. The client has no idea a proxy is involved — it thinks it's talking directly to the server. The backend servers behind the proxy are hidden from the outside world.

---

=== "Forward Proxy"

    Forward proxies sit on the client side of the connection. The client sends requests to the proxy, which forwards them to the destination server.

    ### Why Use a Forward Proxy?

    **Privacy and anonymity.** The destination server sees the proxy's IP address instead of the client's. VPN services and the Tor network are essentially forward proxy systems. Corporate employees browsing through a company proxy appear to external servers as the company, not as individuals.

    **Access control.** Corporate networks use forward proxies to enforce internet usage policies — block certain domains, restrict access to specific content categories, or log all outbound traffic for compliance. Schools and libraries use them to filter content.

    **Caching and bandwidth savings.** When 500 employees all access the same software update or popular website, the forward proxy caches the response after the first request. The remaining 499 requests are served from cache, saving bandwidth and reducing latency. ISPs have historically used transparent forward proxies for this purpose.

    **Bypassing geographic restrictions.** A user in Country A routes traffic through a forward proxy in Country B to access content restricted to Country B. This is the core use case for consumer VPN services.

    ### Transparent vs Explicit Proxies

    An **explicit proxy** requires the client to be configured to use it — the browser or application sends requests to the proxy's address. An **transparent proxy** intercepts traffic at the network level without any client configuration. The client doesn't know the proxy exists. ISPs and corporate networks use transparent proxies to cache content or filter traffic without requiring every device to be configured.

=== "Reverse Proxy"

    Reverse proxies are far more common in system design discussions because they solve server-side infrastructure problems. Nginx, HAProxy, Envoy, Cloudflare, and AWS ALB are all reverse proxies.

    ### Pattern 1: Load Distribution

    The most common reverse proxy use case. Route incoming requests across multiple backend servers to distribute load and provide redundancy. See [Load Balancers](load-balancers.md) for a deep dive on algorithms and strategies.

    ```
                         ┌──→ App Server 1
    Client ──→ Nginx ────┼──→ App Server 2
                         └──→ App Server 3
    ```

    ### Pattern 2: SSL Termination

    HTTPS encryption is computationally expensive. Instead of every backend server handling SSL handshakes, the reverse proxy terminates SSL — it decrypts incoming HTTPS requests, then forwards plain HTTP to backend servers on the internal network.

    ```
    Client ──HTTPS──→ Reverse Proxy ──HTTP──→ Backend
                           │
                      Handles TLS:
                      - Certificate management
                      - Handshake processing
                      - Encryption/decryption
    ```

    This centralizes certificate management (one place to renew certs, one place to configure TLS versions and cipher suites) and offloads CPU-intensive cryptographic work from application servers. **Netflix** terminates SSL at their edge proxies, handling hundreds of millions of TLS handshakes per day before forwarding decrypted traffic to backend services.

    ### Pattern 3: Caching

    The reverse proxy caches responses from backend servers. When the same content is requested again, the proxy serves it from cache without bothering the backend.

    ```
    Request ──→ Reverse Proxy
                    │
                    ├─ Cache HIT → serve directly (1-2ms)
                    │
                    └─ Cache MISS → forward to backend (50-200ms)
                                         │
                                    store response in cache
    ```

    This is particularly effective for:
    - **Static assets** (images, CSS, JavaScript) — cache for hours or days
    - **API responses** that don't change frequently — cache with short TTLs (30-300 seconds)
    - **Pages that are expensive to generate** — render once, serve from cache thousands of times

    **Varnish** is a dedicated caching reverse proxy used by Wikipedia, The Guardian, and many high-traffic news sites.

    ### Pattern 4: API Gateway

    In a microservices architecture, the reverse proxy becomes the single entry point for all client requests, routing them to the appropriate service based on the URL path, headers, or other request attributes.

    ```
                              ┌──→ /users/*    → User Service
    Client ──→ API Gateway ───┼──→ /orders/*   → Order Service
                              ├──→ /products/* → Product Service
                              └──→ /search/*   → Search Service
    ```

    Beyond routing, an API gateway typically handles:
    - **Authentication:** Verify tokens before forwarding to services
    - **Rate limiting:** Protect services from traffic spikes
    - **Request/response transformation:** Convert between protocols, add headers
    - **Response aggregation:** Combine responses from multiple services into one

    **Kong**, **AWS API Gateway**, and **Envoy** are popular API gateway implementations. Netflix's **Zuul** gateway handles billions of requests per day, routing to hundreds of microservices.

=== "Service Mesh"

    ### Sidecar Proxy

    In a microservices architecture, every service talks to many other services. Managing load balancing, retries, timeouts, encryption, and observability for each connection is a cross-cutting concern that shouldn't be in business logic.

    A **service mesh** solves this by deploying a proxy sidecar alongside every service instance. All inter-service traffic flows through these sidecar proxies, which handle networking concerns transparently.

    ```
    ┌─────────────────────┐         ┌─────────────────────┐
    │  Service A          │         │  Service B          │
    │  ┌───────────────┐  │         │  ┌───────────────┐  │
    │  │  App Code     │  │         │  │  App Code     │  │
    │  └───────┬───────┘  │         │  └───────┬───────┘  │
    │          │          │         │          │          │
    │  ┌───────▼───────┐  │         │  ┌───────▼───────┐  │
    │  │ Sidecar Proxy │◄─┼─────────┼─►│ Sidecar Proxy │  │
    │  │   (Envoy)     │  │  mTLS   │  │   (Envoy)     │  │
    │  └───────────────┘  │         │  └───────────────┘  │
    └─────────────────────┘         └─────────────────────┘
    ```

    The sidecar proxy handles:
    - **Mutual TLS (mTLS):** Encrypt all service-to-service traffic without application changes
    - **Load balancing:** Client-side load balancing with circuit breaking
    - **Retries and timeouts:** Configurable per-route retry policies
    - **Observability:** Distributed tracing, metrics, and access logs for every request

    **Istio** (using Envoy sidecars) is the most widely adopted service mesh. **Linkerd** is a lighter alternative. **Lyft** built Envoy specifically for this use case when they migrated to microservices — it now handles millions of requests per second across their infrastructure.

=== "Edge Proxy"

    An edge proxy sits at the perimeter of your network, handling traffic as it enters from the public internet. It's the first thing external requests hit and the last thing responses pass through.

    ```
    Internet                        Your Infrastructure
    ─────────                       ────────────────────
    Users ──→ Edge Proxy ──→ Regional LB ──→ Backend Services
                   │
              At the edge:
              - DDoS mitigation
              - Bot detection
              - Geographic routing
              - TLS termination
              - WAF (attack filtering)
              - Compression
    ```

    **Cloudflare** operates the world's largest edge proxy network with 300+ data centers globally. When you put your site behind Cloudflare, every request first hits a Cloudflare edge server near the user. It filters attacks, caches content, and only forwards legitimate requests to your origin servers.

    **Fastly** and **AWS CloudFront** serve similar roles, combining edge proxying with CDN functionality.

    The key insight: edge proxies protect your infrastructure by filtering traffic *before* it reaches your servers. A DDoS attack targeting your origin IP hits Cloudflare's distributed network first — they absorb the traffic, and your servers never see it.

---

## Proxy vs Load Balancer vs API Gateway

These terms overlap significantly, which causes confusion. Here's how they relate:

| Concept | Core Function | Example |
|---|---|---|
| **Reverse Proxy** | Sits between clients and servers, forwards requests | Nginx, HAProxy |
| **Load Balancer** | A reverse proxy that distributes across multiple backends | ALB, NLB, HAProxy |
| **API Gateway** | A reverse proxy with auth, rate limiting, routing for APIs | Kong, AWS API Gateway |
| **Service Mesh** | Proxies between microservices (sidecar pattern) | Istio, Linkerd |
| **CDN** | Reverse proxy with caching at geographically distributed edges | Cloudflare, CloudFront |

A load balancer **is** a reverse proxy. An API gateway **is** a reverse proxy. A CDN **is** a reverse proxy with caching. They're all specializations of the same fundamental pattern: intercepting traffic between clients and servers.

---

## Key Takeaways

1. **Forward proxies represent clients, reverse proxies represent servers.** The distinction is about which side of the connection the proxy works for.

2. **Reverse proxies are the backbone of modern web infrastructure.** SSL termination, load balancing, caching, and API routing all happen at the reverse proxy layer.

3. **Service meshes bring proxy patterns inside the data center.** Instead of one proxy at the edge, every service gets its own sidecar proxy for service-to-service communication.

4. **Edge proxies are your first line of defense.** DDoS protection, WAF, and bot detection happen at the network edge before traffic reaches your infrastructure.

5. **Most "different" infrastructure components are just specialized proxies.** Load balancers, API gateways, CDNs, and service meshes are all variations on the proxy pattern with different features emphasized.

---

## Related Topics

- **[Load Balancers](load-balancers.md)** — algorithms and strategies for distributing traffic
- **[CDN](cdn.md)** — caching reverse proxies at the network edge
- **[Authentication](../security/authentication.md)** — how API gateways handle auth
- **[Fault Tolerance](../reliability/fault-tolerance.md)** — circuit breakers and retries in proxy layers
