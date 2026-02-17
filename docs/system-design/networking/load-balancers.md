# Load Balancers

A load balancer sits between clients and a group of servers, distributing incoming requests so that no single server bears too much load. It's one of the first pieces of infrastructure you add when scaling beyond a single machine — and one of the most important. Without it, a single server failure means total downtime. With it, you get fault tolerance, horizontal scaling, and the ability to deploy without interruption.

This guide covers how load balancers work at different network layers, the algorithms they use to decide where to send traffic, and how real systems put it all together.

---

=== "How Load Balancers Work"

    Every load balancer does the same fundamental thing: accept a connection from a client, choose a backend server, and forward the request. The differences lie in *what information* the load balancer uses to make that decision.

    ### Layer 4 (Transport Layer)

    A Layer 4 load balancer operates at the TCP/UDP level. It sees IP addresses and port numbers, but never inspects the actual content of the request. When a connection arrives, the load balancer picks a backend based on the source/destination IP and port, then forwards all packets in that connection to the chosen server.

    Because L4 load balancers don't parse HTTP headers or URLs, they're extremely fast — they can handle millions of connections per second with minimal latency. AWS Network Load Balancer (NLB) operates at this layer.

    **Best for:** High-throughput applications, non-HTTP protocols (database connections, gRPC, SMTP), and situations where raw performance matters more than routing flexibility.

    **Limitation:** You can't route `/api/*` to one set of servers and `/static/*` to another — L4 doesn't know what URL the client requested.

    ### Layer 7 (Application Layer)

    A Layer 7 load balancer understands HTTP. It reads the request URL, headers, cookies, and even the body before deciding where to route it. This enables powerful routing rules:

    - Send `/api/users` to the user service and `/api/orders` to the order service
    - Route mobile clients (detected by User-Agent header) to a different backend
    - Terminate SSL, inspect the request, then forward over plain HTTP internally
    - Reject requests that match known attack patterns (acting as a basic WAF)

    The trade-off is performance — parsing HTTP adds latency compared to L4. But for most web applications, the routing flexibility is worth the small overhead. Nginx, HAProxy (in HTTP mode), and AWS Application Load Balancer (ALB) all operate at this layer.

    ### DNS Load Balancing

    The simplest form: configure your DNS to return different IP addresses for the same domain name. Each client resolves the domain to a different server. Route 53's weighted routing and latency-based routing are common implementations.

    DNS load balancing is useful for geographic distribution (route European users to EU servers), but it's coarse-grained. DNS records are cached by clients and resolvers, so changes propagate slowly. You can't react to a server going down in real time — stale DNS entries will keep sending traffic to dead servers for minutes or hours.

    **Best as a first layer** that routes to regional load balancers, which then handle fine-grained distribution.

=== "Algorithms"

    The algorithm determines which server gets the next request. The right choice depends on whether your servers are identical, whether requests take uniform time, and whether you need session affinity.

    ### Round Robin

    Send each request to the next server in sequence: server 1, server 2, server 3, back to server 1. Simple, stateless, and perfectly fair when all servers have the same capacity and all requests take roughly the same time to process.

    **Breaks down when:** Servers have different specs (a 32-core machine gets the same traffic as a 4-core one), or request processing times vary widely (one request takes 5ms, another takes 5 seconds — the 5-second request's server accumulates a queue while others sit idle).

    ### Weighted Round Robin

    Same as round robin, but servers with more capacity get proportionally more traffic. Assign a weight to each server based on its resources:

    ```
    Server A (16 cores): weight 4  → receives 4/7 of traffic
    Server B (8 cores):  weight 2  → receives 2/7 of traffic
    Server C (4 cores):  weight 1  → receives 1/7 of traffic
    ```

    Useful during deployments too — route 5% of traffic to a canary server by giving it weight 1 while production servers have weight 19.

    ### Least Connections

    Route each new request to the server with the fewest active connections. This naturally adapts to varying request durations: a server processing a slow request accumulates connections and stops receiving new ones until it catches up.

    **Best for:** Applications with highly variable request times — WebSocket connections, file uploads, database queries that range from 1ms to 10 seconds. GitHub uses least-connections balancing for their API servers because request durations vary dramatically.

    ### IP Hash

    Hash the client's IP address to deterministically map it to a server. The same client always reaches the same server, providing session affinity without cookies or shared session stores.

    ```
    hash(client_ip) % num_servers = server_index

    Client 10.0.0.1  → hash → 7  → 7 % 3 = 1  → Server B
    Client 10.0.0.2  → hash → 12 → 12 % 3 = 0 → Server A
    Client 10.0.0.1  → hash → 7  → 7 % 3 = 1  → Server B  (same server)
    ```

    **The problem:** When you add or remove a server, the modulo changes and most clients get remapped to different servers — breaking all existing sessions. Consistent hashing solves this by only remapping clients that were on the added/removed server.

    ### Least Response Time

    Route to the server that's responding fastest. This combines connection count with actual measured latency, accounting for both load and server performance. If Server A is responding in 5ms and Server B in 50ms, new requests go to Server A.

    More sophisticated than least connections but requires the load balancer to track response times, adding slight overhead.

    ### Algorithm Selection Guide

    | Your Situation | Algorithm | Why |
    |---|---|---|
    | Identical servers, uniform requests | **Round Robin** | Simple, even distribution |
    | Mixed server sizes | **Weighted Round Robin** | Respects capacity differences |
    | Variable request durations | **Least Connections** | Adapts to actual load |
    | Need session affinity | **IP Hash** | Same client hits same server |
    | Mixed sizes + variable durations | **Weighted Least Connections** | Best of both worlds |

=== "Operations"

    ### Health Checks

    A load balancer that sends traffic to dead servers is worse than no load balancer at all. Health checks are how the load balancer knows which backends are alive.

    #### Active Health Checks

    The load balancer periodically sends a probe request to each backend — typically an HTTP GET to `/health`. If a server fails to respond (or returns an error) for a configured number of consecutive attempts, the load balancer removes it from the pool. When the server starts responding again, it's added back.

    Key parameters:
    - **Interval:** How often to check (10-30 seconds is typical)
    - **Timeout:** How long to wait for a response (2-5 seconds)
    - **Unhealthy threshold:** How many consecutive failures before removal (2-3)
    - **Healthy threshold:** How many consecutive successes before re-adding (2-3)

    **Shallow checks** (`GET /health` returns 200 if the process is running) are fast but miss deeper problems. **Deep checks** (`GET /health/ready` verifies database connectivity, disk space, downstream dependencies) catch more issues but take longer and can cascade failures if a shared dependency is slow.

    #### Passive Health Checks

    Instead of sending probes, the load balancer monitors real traffic. If a server starts returning 5xx errors or timing out on actual requests, the load balancer reduces or stops sending traffic to it. This catches issues that active checks might miss — like a server that responds to health checks but fails under real load.

    The best setups use both: active checks for fast detection of complete failures, passive checks for detecting degradation under load.

    ### Load Balancer Technologies

    **HAProxy** — The gold standard for high-performance load balancing. Handles both L4 and L7, supports every algorithm mentioned above, and can manage hundreds of thousands of concurrent connections on modest hardware. Instagram, GitHub, Stack Overflow, and Airbnb all use HAProxy.

    **Nginx** — Started as a web server, now widely used as a reverse proxy and L7 load balancer. Excellent for HTTP-based routing, SSL termination, and serving static files alongside load balancing. Often the simplest choice when you're already using Nginx as your web server.

    **AWS ALB (Application Load Balancer)** — Managed L7 load balancer. Path-based and host-based routing, native integration with ECS/EKS/Lambda, automatic scaling. The default choice for AWS-based applications.

    **AWS NLB (Network Load Balancer)** — Managed L4 load balancer. Handles millions of requests per second with ultra-low latency. Preserves client source IPs. Used for non-HTTP workloads or when you need extreme throughput.

    **Envoy** — Modern L7 proxy built for microservices. Powers the Istio service mesh. Advanced features like circuit breaking, retry budgets, and observability built in. Used by Lyft, Airbnb, and Stripe.

    ### Session Affinity (Sticky Sessions)

    Sometimes you need the same user to reach the same server — for example, if session data is stored in server memory rather than a shared store like Redis.

    **Cookie-based:** The load balancer sets a cookie (like `SERVERID=backend2`) on the first response. Subsequent requests include this cookie, and the load balancer routes accordingly. Most flexible and reliable.

    **IP hash:** As described above — deterministic mapping from client IP to server. Breaks when clients share IPs (corporate NAT) or change IPs (mobile networks).

    **The better solution** in most cases: don't use sticky sessions at all. Store session data in Redis or a database, and let the load balancer distribute freely. Sticky sessions reduce the load balancer's ability to distribute evenly, and when the "sticky" server goes down, the user's session is lost anyway.

    ### High Availability for the Load Balancer Itself

    A load balancer that's a single point of failure defeats the purpose. Production setups use redundancy:

    **Active-passive:** Two load balancers share a virtual IP (VIP). The active one handles all traffic. If it fails, the passive one takes over the VIP within seconds (using VRRP or similar). Simple and common.

    **Active-active:** Multiple load balancers handle traffic simultaneously, typically behind DNS round robin or anycast. Higher throughput and no wasted standby capacity, but more complex to configure.

    Cloud managed load balancers (ALB, NLB, Google Cloud Load Balancing) handle this automatically — they're distributed across multiple availability zones with built-in redundancy.

---

## Key Takeaways

1. **Layer 4 for speed, Layer 7 for flexibility.** Use L4 when you need raw throughput or non-HTTP protocols. Use L7 when you need content-based routing.

2. **Least connections is the safest default** for most web applications with variable request times.

3. **Health checks are not optional.** Use both active probes and passive monitoring. A load balancer without health checks is just a traffic splitter.

4. **Avoid sticky sessions when possible.** Externalize session state to Redis or a database for better fault tolerance and load distribution.

5. **The load balancer must be redundant too.** Active-passive pairs, active-active clusters, or managed cloud services.

6. **Start with managed services.** AWS ALB, Google Cloud Load Balancing, or Cloudflare handle scaling, health checks, and redundancy for you. Roll your own with HAProxy or Nginx only when you need specific control.

---

## Related Topics

- **[Proxies](proxies.md)** — reverse proxies, forward proxies, and the patterns they enable
- **[CDN](cdn.md)** — distributing content globally via edge servers
- **[Horizontal Scaling](../scalability/horizontal-scaling.md)** — adding servers behind a load balancer
- **[Fault Tolerance](../reliability/fault-tolerance.md)** — designing systems that survive failures
