# Networking Fundamentals

The internet is not magic — it's packets finding paths through routers, getting temporarily stored in buffers, and occasionally getting lost. As a software engineer, you don't need to memorize the OSI model or know every TCP flag, but you do need to understand *why* a request to a nearby server sometimes takes 200ms, why increasing your connection pool size doesn't always improve throughput, and what actually happens when a network call fails.

This guide covers the networking fundamentals that directly impact how you design and debug systems: latency budgets, connection management, network failures, and the minimal theory that ties it all together.

---

## How the Internet Actually Works

When your application sends an HTTP request, here's what happens at the network level. Your operating system breaks the data into **packets** — small chunks, typically 1,500 bytes each. Each packet gets an IP header (the destination address) and a TCP header (sequence numbers for reassembly). These packets leave your machine and enter the network.

```
Your App (California)                              Server (London)
    │                                                   │
    ▼                                                   ▼
[OS Network Stack]                              [OS Network Stack]
    │                                                   ▲
    ▼                                                   │
Router → Router → Router → Router → Router → Router → Router
 (1ms)   (2ms)    (5ms)   (50ms)   (10ms)    (5ms)   (2ms)

One-way latency: ~75ms (speed of light through fiber + routing)
Round trip: ~150ms (request + response)
```

Each router along the path examines the packet's destination IP address and forwards it to the next router that's closer to the destination. This process — called **routing** — happens at every hop. The internet has no central controller; each router makes its own forwarding decision based on routing tables maintained by protocols like BGP (Border Gateway Protocol).

The key insight for software engineers: **you can't beat physics.** Light through fiber travels at roughly 200,000 km/s. California to London is about 8,500 km, so the absolute minimum one-way latency is ~43ms. Real-world latency is 2-3x worse because of routing detours, processing at each hop, and queuing in buffers. This is why CDNs and multi-region deployments matter — they reduce the physical distance between users and servers.

**Private vs public IP addresses.** Inside your VPC or data center, servers use private IPs (10.x.x.x, 172.16-31.x.x, 192.168.x.x) that don't route over the public internet. A NAT gateway translates between private and public IPs at the network boundary. This is why two servers in the same availability zone communicate in <1ms while the same request over the public internet takes 10-50ms.

---

## Latency Numbers Every Programmer Should Know

These numbers shape every architecture decision. Memorize the orders of magnitude, not the exact values.

```
Operation                                    Time         Notes
─────────────────────────────────────────────────────────────────
L1 cache reference                           0.5 ns
L2 cache reference                           7 ns
Main memory reference                        100 ns
SSD random read                              16 us       (16,000 ns)
Read 1 MB sequentially from SSD              0.2 ms
Round trip within same datacenter             0.5 ms      (500 us)
Read 1 MB sequentially from network           1 ms
Disk seek (HDD)                               2 ms
Read 1 MB sequentially from HDD               5 ms
Round trip: US East to US West               40 ms
Round trip: US to Europe                     80-150 ms
Round trip: US to Asia                       150-300 ms
```

The critical takeaway: **network latency dominates everything in distributed systems.** A memory lookup takes 100 nanoseconds. A cross-continent network round trip takes 150 milliseconds — that's 1.5 million times slower. This is why caching, data locality, and reducing network round trips are the highest-impact optimizations.

### Latency Budget for a Typical API Call

```
Total P99 latency target: 200ms

DNS resolution:          1ms   (cached -- first call is 20-50ms)
TCP handshake:          40ms   (one round trip)
TLS handshake:          40ms   (one round trip for TLS 1.3)
HTTP request transit:    2ms   (small payload)
Server processing:      50ms   (database query + logic)
HTTP response transit:  10ms   (5KB JSON response)
                       -----
                       143ms   (57ms budget remaining)
```

**Amazon** found that every 100ms of latency costs 1% in sales. **Google** found that an extra 500ms in search page load time caused a 20% drop in traffic. These numbers haven't changed — if anything, users have become less patient. Design your systems with a latency budget and track where every millisecond goes.

---

## The OSI Model -- Only the Layers That Matter

The OSI model has 7 layers. For software engineering, you really need to understand 2 of them.

| Layer | Name | What It Does | Why You Care |
|---|---|---|---|
| **Layer 7** | Application | HTTP, WebSocket, gRPC, DNS | This is where you live. Protocol choices, API design, request/response patterns |
| **Layer 4** | Transport | TCP, UDP | Reliability vs speed trade-off. Connection management, port numbers |
| Layers 1-3 | Physical, Data Link, Network | Cables, switches, IP routing | "The network team handles this." You rarely interact directly |
| Layers 5-6 | Session, Presentation | Session management, encoding | Mostly absorbed into Layer 7 in modern protocols |

**Layer 4 (Transport)** is where the reliability decision happens. TCP guarantees delivery, ordering, and no duplicates — at the cost of a 3-way handshake and retransmission delays. UDP sends packets with no guarantees — faster, but your application must handle loss. This choice cascades through your entire architecture. See [Protocols](../networking/protocols.md) for when to choose each.

**Layer 7 (Application)** is where you make most engineering decisions. Which HTTP version? REST or gRPC? WebSocket or Server-Sent Events? These choices affect latency, throughput, browser compatibility, and developer experience. See [Protocols](../networking/protocols.md) and [API Design](../communication/api-design/index.md) for detailed comparisons.

---

## Connection Management

How your application manages network connections has a direct impact on performance and resource usage. There are four approaches, from simplest to most efficient.

### Short-Lived Connections

Open a connection, send one request, close the connection. This was the default in HTTP/1.0. Every request pays the full cost: TCP handshake (1 round trip), TLS handshake (1 round trip), then the actual request.

For a server 40ms away, that's 80ms of overhead before a single byte of useful data transfers. With 100 requests per page load, that's 8 seconds of pure connection overhead. Nobody uses this anymore for a reason.

### Persistent Connections (Keep-Alive)

HTTP/1.1 introduced `Connection: keep-alive` — after the response, the TCP connection stays open for reuse. The next request skips the handshake entirely. The server typically closes idle connections after 60-120 seconds.

This is the default in all modern HTTP clients and saves 40-80ms per request after the first one. The limitation: HTTP/1.1 can only send one request at a time per connection, so browsers open 6 parallel connections per domain.

### Connection Pooling

Pre-establish a pool of connections and reuse them across requests. This is critical for database connections, where creating a new connection involves authentication, session setup, and memory allocation — often 50-100ms.

```
Pool: min=5, max=20, idle timeout=5min

Idle:     [*****               ]  5 connections ready (min pool)
Busy:     [********************]  20 connections active (max pool)
Draining: [**********          ]  Traffic dropped, excess closing

* = connection    Each request borrows a connection,
                  uses it, returns it to the pool.
```

**Pool sizing matters.** Too small and requests queue waiting for a connection. Too large and you exhaust database connection limits (PostgreSQL defaults to 100 max connections). A good starting point: **pool size = number of concurrent requests your service handles**. For most web applications, min=5 and max=20 covers typical workloads. Use PgBouncer for PostgreSQL or HikariCP for Java to manage pools efficiently.

**Netflix** tunes connection pools per downstream service — high-traffic services get larger pools, low-traffic services share smaller pools. They found that right-sizing connection pools reduced P99 latency by 30% compared to a one-size-fits-all configuration.

### Multiplexing

HTTP/2 and gRPC go further: multiple requests share a single connection simultaneously. Request A and Request B interleave their data frames on the same TCP connection. No head-of-line blocking at the HTTP layer, no need for multiple connections.

This is the most efficient approach. One connection handles hundreds of concurrent requests, with minimal overhead per request. The trade-off: a single connection means a single TCP stream, so a lost packet blocks all requests until retransmission completes (TCP head-of-line blocking). HTTP/3 solves this by running on UDP.

---

## Network Failures and How to Handle Them

Networks fail. Not occasionally — constantly. Packets get dropped, servers crash mid-response, connections time out. The question isn't whether your system will experience network failures, but whether it handles them gracefully.

### Common Failure Modes

| Failure | What Happened | What You See |
|---|---|---|
| **Connection refused** | Server process is down or not listening on that port | Immediate error, no delay |
| **Connection timeout** | Network is blocking traffic (firewall, routing issue) | Hangs for timeout duration, then fails |
| **Connection reset** | Server crashed mid-request, or load balancer killed connection | Partial response or abrupt error |
| **Read timeout** | Server is alive but processing too slowly | Hangs, then fails after read timeout |
| **DNS failure** | DNS server unreachable or domain doesn't exist | Fails before connection attempt |

### Handling Strategies

**Timeouts.** Every network call needs a timeout. Without one, a slow server blocks your thread indefinitely. Set connection timeouts (how long to wait for the TCP handshake) and read timeouts (how long to wait for data after connecting) separately. Typical values: 5 seconds for connection, 30 seconds for read — but tune based on your actual P99 latencies.

**Retries with backoff.** Transient failures — brief network blips, server restarts, momentary overload — often resolve on their own. Retry with exponential backoff: wait 1 second, then 2, then 4, then 8. Add jitter (randomness) so that 1,000 clients don't all retry at the same instant. Cap at 3-5 retries.

**Idempotency.** Safe to retry a GET request — reading the same data twice is harmless. But retrying a POST that creates an order might create two orders. For non-idempotent operations, use **idempotency keys**: the client sends a unique request ID, and the server deduplicates. **Stripe** uses this pattern for all payment APIs — the same idempotency key always returns the same result, making retries safe.

**Circuit breakers.** If a downstream service is failing consistently, retrying every request wastes resources and adds latency. A circuit breaker detects the failure pattern (e.g., 50% of requests failing in the last 30 seconds) and stops making calls entirely for a cooldown period. See [Fault Tolerance](../reliability/fault-tolerance.md) for the full pattern.

---

## Bandwidth vs Latency vs Throughput

These three concepts are frequently confused, and the confusion leads to wrong optimization choices.

**Bandwidth** is the capacity of the pipe — how much data can flow per second. A 1 Gbps link can theoretically transfer 125 MB/s. For most web applications, bandwidth is rarely the bottleneck. A typical API response is 5-50 KB; even at 10,000 requests per second, that's only 50-500 MB/s.

**Latency** is the time for a single packet to travel from source to destination. It's dominated by physical distance and cannot be reduced by buying bigger servers. This is the bottleneck for most applications — every network round trip adds latency, and user experience degrades linearly with it.

**Throughput** is the actual data transferred per unit of time. It's limited by both bandwidth and latency. A high-bandwidth link with high latency has surprisingly low throughput because TCP's flow control requires acknowledgments before sending more data.

```
Scenario: 1 Gbps link, 100ms round trip

Theoretical max:  1 Gbps = 125 MB/s
Actual with TCP:  ~12 MB/s

Why? TCP can only have ~1.5 MB "in flight" before needing an ACK.
     1.5 MB / 0.1s = 15 MB/s maximum, regardless of bandwidth.

Fix: TCP window scaling, or multiple parallel connections.
```

The practical lesson: **optimize for latency first, bandwidth second.** Reduce network round trips (batch requests, use HTTP/2 multiplexing, cache aggressively). Only worry about bandwidth when transferring large files or streaming video.

---

## Key Takeaways

1. **Network latency dominates distributed system performance.** A cross-continent round trip (150ms) is 1.5 million times slower than a memory access. Reducing round trips is the highest-impact optimization.

2. **Know your latency budget.** DNS + TCP + TLS + transit + processing. Track where every millisecond goes and optimize the biggest contributors first.

3. **Connection pooling is essential for databases.** Creating connections is expensive. Pool them, size them appropriately (min=5, max=20 is a good start), and monitor utilization.

4. **Every network call needs a timeout.** Without timeouts, a slow dependency will eventually consume all your threads and bring down your service.

5. **Retry with backoff and idempotency keys.** Transient failures are normal. Retries recover from them — but only when combined with exponential backoff (to prevent thundering herds) and idempotency (to prevent duplicate side effects).

6. **You can't beat physics.** Light through fiber has a speed limit. CDNs, multi-region deployments, and edge computing reduce latency by reducing distance — not by going faster.

---

## Related Topics

- **[Protocols](../networking/protocols.md)** — TCP vs UDP, HTTP versions, WebSocket vs SSE
- **[DNS](../networking/dns.md)** — how domain resolution works
- **[Load Balancers](../networking/load-balancers.md)** — distributing traffic across servers
- **[Fault Tolerance](../reliability/fault-tolerance.md)** — circuit breakers, retries, and graceful degradation
- **[CDN](../networking/cdn.md)** — reducing latency through edge caching
