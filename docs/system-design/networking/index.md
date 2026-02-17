# Networking

Networking is the foundation of every distributed system. As a software engineer, you don't need to memorize TCP flags or BGP routing tables — but you do need to understand how DNS propagation affects your deployments, when to choose WebSocket over HTTP, and why a CDN reduces latency by 80%.

This section focuses on practical networking knowledge: the concepts that directly impact your architecture decisions.

---

## Topics

| Topic | What You'll Learn | When You Need It |
|---|---|---|
| **[DNS](dns.md)** | How domain resolution works, record types, Route53 routing strategies | Setting up domains, multi-region deployments, debugging "DNS not propagated yet" |
| **[Protocols](protocols.md)** | TCP vs UDP, HTTP/1.1 vs 2 vs 3, WebSocket vs SSE vs long polling | Choosing communication patterns for your architecture |
| **[Load Balancers](load-balancers.md)** | L4 vs L7 load balancing, algorithms, health checks | Scaling beyond a single server |
| **[Proxies](proxies.md)** | Forward vs reverse proxy, SSL termination, API gateway patterns | Adding security, caching, or routing layers |
| **[CDN](cdn.md)** | Edge caching, cache invalidation, global content distribution | Serving static assets, reducing latency for global users |

---

## Where to Start

**Just starting with distributed systems?** Start with [Load Balancers](load-balancers.md) — it's the first infrastructure you add when scaling beyond one server.

**Deploying to production?** Read [DNS](dns.md) — understand TTL and propagation before your first deployment catches you off guard.

**Optimizing performance?** Start with [CDN](cdn.md) — the biggest latency wins come from serving content closer to users.

**Choosing how services communicate?** Read [Protocols](protocols.md), then dive into [API Design](../communication/api-design/index.md) for REST vs GraphQL vs gRPC.

**Adding a security or caching layer?** Learn about [Proxies](proxies.md) — reverse proxies handle SSL termination, caching, and request routing.

---

## Related Sections

- **[Networking Fundamentals](../fundamentals/networking-fundamentals.md)** — how the internet works, latency numbers, connection management
- **[API Design](../communication/api-design/index.md)** — REST, GraphQL, gRPC, and WebSocket patterns in depth
- **[Security](../security/index.md)** — TLS, certificates, and network security
- **[Scalability Patterns](../scalability/patterns.md)** — how networking fits into the broader scaling journey
