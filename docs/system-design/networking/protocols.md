# Network Protocols

Every client-server communication starts with a choice: which protocol should carry this data? HTTP handles your REST APIs, WebSocket powers your chat features, gRPC connects your microservices, and UDP streams your video calls. Each protocol makes different trade-offs between reliability, latency, complexity, and compatibility. Most engineers default to HTTP for everything — and that works surprisingly well — but understanding when to reach for something else is the difference between a system that merely functions and one that performs.

This guide focuses on the practical question: "which protocol should I use for this feature?" Rather than cataloging every protocol in existence, it covers the ones you will actually encounter when designing systems at scale -- and the trade-offs that determine which to pick.

```
Need to communicate between client and server?
    |
    |-- Request-response (most APIs) -----------> HTTP/2 (default) or REST
    |
    |-- Real-time bidirectional ----------------> WebSocket
    |
    |-- Server push only -----------------------> Server-Sent Events (SSE)
    |
    |-- Internal microservices, high throughput -> gRPC
    |
    '-- Video / voice / gaming -----------------> UDP-based (WebRTC)
```

---

=== "TCP vs UDP"

    This section is not about how TCP and UDP work internally — it is about when you would choose one over the other, and why that choice ripples through every layer above it.

    ### TCP: Reliability First

    TCP guarantees that every byte you send arrives at the other end, in order, exactly once. It achieves this through a three-way handshake (SYN, SYN-ACK, ACK) that costs one full round trip before any data flows, plus sequence numbers and acknowledgments that track every packet. If a packet is lost, TCP detects the gap and retransmits it. If packets arrive out of order, TCP reassembles them before handing data to the application.

    This makes TCP the right choice when data integrity matters more than speed. Every HTTP request (through HTTP/2) runs on TCP. Database connections use TCP because a single corrupted query could drop a table. File transfers use TCP because one missing byte corrupts the entire file. Payment processing uses TCP because a lost acknowledgment could mean charging a customer twice — or not at all. The common thread: any scenario where "close enough" is not good enough requires TCP.

    The cost is latency. That initial handshake adds one round trip (50-100ms across continents) before the first byte of useful data. Add TLS on top and you are looking at two or three round trips just to establish a secure connection — that is 150-300ms before a single byte of application data moves.

    ```
    TCP + TLS connection setup:

    Client                          Server
      |── SYN ────────────────────>   |     ┐
      |<──────────────── SYN-ACK ──   |     ├─ TCP handshake (1 RTT)
      |── ACK ────────────────────>   |     ┘
      |── ClientHello ────────────>   |     ┐
      |<──────────── ServerHello ──   |     ├─ TLS handshake (1-2 RTT)
      |── Finished ───────────────>   |     ┘
      |── GET /api/data ──────────>   |     first useful data
    ```

    ### UDP: Speed First

    UDP takes the opposite approach: send the packet and move on. No handshake, no acknowledgments, no retransmission. A UDP packet either arrives or it does not, and the sender never finds out either way. This makes UDP fundamentally faster — there is no connection setup delay, and no waiting for lost-packet recovery.

    The trade-off is that the application must handle unreliability itself. If you are building a video call, a dropped frame from 200ms ago is worthless — the conversation has moved on. Retransmitting it would only add latency and show the viewer stale imagery. But if you are building a file transfer, you need every byte, in order. UDP gives you the freedom to decide what "reliable enough" means for your specific use case.

    DNS queries use UDP because each query is a single small packet that either gets a response or gets retried by the application. Video streaming, online gaming, and voice calls use UDP because fresh data matters more than complete data. A multiplayer game running at 60 frames per second sends positional updates via UDP — if one update is lost, the next one arrives 16ms later with newer information anyway.

    ### When TCP Becomes the Bottleneck

    Consider a video streaming service serving 4K content. Each second of 4K video is roughly 15-25 megabits. If a single TCP packet is lost, TCP's congestion control algorithm slows the entire connection — not just the lost packet's stream — potentially causing the video to buffer. The user sees a spinner, and a study by **Akamai** found that a 2-second delay in video start time increases abandonment by 6%. This is why Netflix, YouTube, and Twitch all use adaptive bitrate streaming over HTTP (which runs on TCP) but are actively migrating to QUIC/HTTP/3, which eliminates this connection-level stalling.

    For truly latency-sensitive applications like online gaming, even QUIC's per-stream recovery is too slow. Games like Fortnite and Valorant use raw UDP with custom application-layer logic: the game client predicts movement locally, sends updates via UDP, and reconciles with the server's authoritative state when packets arrive — a technique called client-side prediction with server reconciliation.

    ### Why QUIC Chose UDP

    HTTP/3 is built on QUIC, which runs over UDP instead of TCP. This seems counterintuitive — why would a protocol that needs reliability choose an unreliable transport?

    The answer is head-of-line blocking. In TCP, if one packet is lost, the entire connection stalls until that packet is retransmitted — even if the lost packet belongs to a different HTTP request multiplexed on the same connection. Imagine loading a web page with 20 resources over HTTP/2: one lost packet for a low-priority analytics script blocks the delivery of your critical CSS and JavaScript.

    QUIC implements its own reliability on top of UDP, but does so per-stream. A lost packet in stream A only blocks stream A; streams B and C continue flowing. QUIC also supports 0-RTT connection resumption for repeat visits, eliminating the handshake overhead entirely for returning clients.

    There is a practical reason QUIC was built on UDP rather than as an entirely new transport protocol: deploying a new transport protocol requires updates to every router, firewall, and middlebox on the internet — a process that takes decades. UDP is already universally supported, so QUIC gets global deployability immediately while implementing its own congestion control, flow control, and reliability in userspace.

    ### Comparison

    | | TCP | UDP | QUIC (HTTP/3) |
    |---|---|---|---|
    | **Reliability** | Guaranteed delivery and ordering | No guarantees | Per-stream reliability |
    | **Connection setup** | 1 RTT (+ 1-2 for TLS) | None | 0-1 RTT |
    | **Head-of-line blocking** | Yes (connection-level) | N/A | No (stream-level) |
    | **Overhead per packet** | 20 bytes | 8 bytes | ~20 bytes |
    | **Best for** | APIs, databases, file transfer | Video, gaming, DNS | Modern web, mobile apps |

    **Zoom** uses UDP for all video and audio streams because tolerating occasional dropped frames is far better than the stuttering caused by TCP retransmissions. When corporate firewalls block UDP (which happens frequently in enterprise environments), Zoom falls back to TCP — and users notice the quality degradation immediately. This dual-protocol approach is common: design for UDP performance, but always have a TCP fallback.

    **Discord** faces the same trade-off for voice chat. Their infrastructure uses UDP for voice data between clients and their media servers, processing over 2.6 million concurrent voice connections. When a user's network blocks UDP, Discord detects the failure within seconds and transparently switches to TCP-based voice, accepting the higher latency to maintain connectivity.

=== "HTTP Evolution"

    HTTP is the protocol that runs the web, and it has evolved dramatically over two decades to keep up with how we build applications. Understanding the differences between versions helps you make informed decisions about performance, compatibility, and infrastructure.

    ### HTTP/1.1: The Workhorse

    Released in 1999 and still carrying a large share of internet traffic, HTTP/1.1 has one fundamental limitation: each TCP connection processes one request at a time. Send a request, wait for the response, then send the next request. This head-of-line blocking means that a slow response (a large image, a heavy database query) delays everything queued behind it.

    Browsers work around this by opening six parallel connections per domain — an ugly but effective hack that trades network efficiency for parallelism. Text-based headers add another cost: every request carries 500-800 bytes of headers, many of them identical across requests (cookies, user-agent, accept headers). On a page that makes 50 requests, that is 25-40KB of redundant header data riding back and forth.

    Developers also used domain sharding as a workaround — serving images from `img1.example.com`, `img2.example.com`, etc. — to bypass the six-connections-per-domain limit. This trick is counterproductive under HTTP/2 (where one connection handles everything), so if you see domain sharding in a legacy codebase, removing it is a free performance win after upgrading.

    Despite these limitations, HTTP/1.1 works everywhere. Every server, proxy, CDN, and firewall on the internet understands it. When compatibility matters above all else — legacy systems, embedded devices, networks with deep packet inspection — HTTP/1.1 is still the safe choice.

    ### HTTP/2: The Modern Default

    HTTP/2 solves HTTP/1.1's biggest problem with multiplexing: many requests and responses flow simultaneously over a single TCP connection, interleaved as binary frames. A slow response no longer blocks other requests at the HTTP layer (though TCP-level head-of-line blocking remains — more on that in the HTTP/3 section).

    Headers are compressed using HPACK, which maintains a lookup table of previously sent headers and transmits only the differences. That 500-byte header that repeated identically on every request now costs 5-10 bytes after the first one. For applications making dozens of API calls per page load, header compression alone can reduce bandwidth consumption by 30-50%.

    HTTP/2 also introduced server push — the ability for the server to proactively send resources before the client requests them. In practice, server push has been largely abandoned (Chrome removed support in 2022). It is difficult to get right because the server cannot know what the client already has cached, pushing already-cached resources wastes bandwidth, and the performance gains proved marginal compared to simpler optimizations like `<link rel="preload">` hints that let the browser decide what it actually needs.

    **Google** reported a 30% reduction in page load latency after migrating their services to HTTP/2, primarily from multiplexing and header compression. **Akamai** found that HTTP/2 reduced page load times by 50-70% for image-heavy sites because all those small image requests could fly over a single connection simultaneously rather than queueing behind each other.

    For most modern web applications, HTTP/2 is the right default — all major browsers, web servers (Nginx, Apache, Caddy), and CDNs support it. The performance improvement over HTTP/1.1 requires zero application code changes; you simply enable it at the server or load balancer level.

    ### HTTP/3 (QUIC): The Future

    HTTP/3 replaces TCP with QUIC (built on UDP), solving the last major performance bottleneck: TCP-level head-of-line blocking. When a packet is lost on an HTTP/2 connection, every multiplexed stream on that connection stalls while TCP retransmits. With HTTP/3, only the affected stream pauses while others continue uninterrupted.

    Two other features make HTTP/3 especially compelling for mobile applications. First, 0-RTT connection resumption lets returning clients send data immediately without any handshake — critical for mobile users who frequently switch between apps and return seconds later. Second, connection migration means a connection survives network changes. When your phone switches from WiFi to cellular while walking out of a coffee shop, HTTP/2 drops the TCP connection entirely and starts from scratch. HTTP/3 continues seamlessly because QUIC identifies connections by a connection ID, not by the IP address and port tuple.

    **YouTube** and **Facebook** serve over 75% of their traffic over HTTP/3. Google measured a 2% reduction in video rebuffering on YouTube after enabling QUIC. These numbers sound small, but at YouTube's scale of billions of daily views, that is millions fewer interrupted viewing sessions per day.

    Adoption is accelerating rapidly. **Cloudflare** enables HTTP/3 by default for all customers, meaning any site behind Cloudflare automatically speaks QUIC to clients that support it — and all modern browsers do. If you use a CDN that supports HTTP/3, you get the benefits without changing a single line of server code.

    ### Request Timeline Comparison

    ```
    HTTP/1.1 (sequential on one connection):

    Client ──GET /style.css──────────────────> Server
           <────────────── response ──────────
           ──GET /app.js────────────────────->
           <────────────── response ──────────
           ──GET /data.json──────────────────>
           <────────────── response ──────────
    Total: 3x round-trip time

    HTTP/2 (multiplexed on one connection):

    Client ──GET /style.css──┐
           ──GET /app.js─────┤ all sent at once
           ──GET /data.json──┘
           <── style.css ──┐
           <── data.json ──┤ responses interleaved
           <── app.js ─────┘
    Total: ~1x round-trip time
    ```

    ### HTTP Version Selection

    | Your Situation | Version | Why |
    |---|---|---|
    | Legacy systems, maximum compatibility | **HTTP/1.1** | Universal support, simplest debugging |
    | Modern web apps, APIs (most cases) | **HTTP/2** | Multiplexing + compression, zero code changes |
    | Mobile-heavy, unreliable networks | **HTTP/3** | Connection migration, 0-RTT, no HoL blocking |

    In practice, you rarely choose a version explicitly. Your web server and CDN negotiate the best version the client supports automatically. The important thing is to ensure your infrastructure (load balancers, proxies, CDNs) supports HTTP/2 at minimum, and HTTP/3 if your audience is mobile-heavy.

=== "Real-Time Communication"

    HTTP is fundamentally pull-based: the client asks, the server answers, the connection closes. This works perfectly for loading web pages and fetching API data, but many features need the server to push data to the client the moment something happens — a new chat message, a stock price change, a teammate's cursor moving across the document. Polling every second is wasteful and still not truly real-time. Three main approaches solve this problem, each with different trade-offs.

    ### WebSocket

    WebSocket starts as a regular HTTP request with an `Upgrade` header. If the server agrees, the connection "upgrades" from HTTP to a persistent, full-duplex channel where both sides can send messages at any time. Each WebSocket frame carries only 2-10 bytes of framing overhead compared to 500+ bytes for a fresh HTTP request.

    ```
    WebSocket lifecycle:

    Client                              Server
      |── GET / (Upgrade: websocket) ──>  |    HTTP handshake
      |<── 101 Switching Protocols ─────  |
      |                                   |
      |══════ persistent connection ══════════
      |                                   |
      |── "hello" ─────────────────────>  |    ┐
      |<───────────── "hi there" ───────  |    │ Full-duplex:
      |<──────── "new message from Bob" ─ |    │ either side sends
      |── "typing..." ────────────────->  |    │ at any time
      |<──────── "Bob is typing..." ────  |    ┘
      |                                   |
      |── close ───────────────────────>  |    Clean shutdown
    ```

    This makes WebSocket the right choice for features that need true bidirectional communication: chat applications, multiplayer games, collaborative editing, and live dashboards where users also send frequent updates. The persistent connection means messages arrive in milliseconds rather than being delayed by HTTP request-response cycles.

    The main limitation is infrastructure compatibility. Some corporate proxies, older load balancers, and aggressive firewalls do not handle WebSocket connections correctly — they may close "idle" connections after 60 seconds or fail to forward the upgrade handshake. Production WebSocket deployments almost always need a fallback mechanism (typically long polling) for these environments.

    Scaling WebSocket also requires architectural consideration. Because connections are persistent and stateful, you cannot simply round-robin requests across servers the way you would with HTTP. If a user is connected to server A and a message for them arrives at server B, you need a pub/sub layer (like Redis Pub/Sub or a message broker) to route that message to the right server. This is fundamentally different from scaling stateless HTTP APIs, where any server can handle any request.

    ### Server-Sent Events (SSE)

    When data only needs to flow in one direction — server to client — SSE is simpler and more robust than WebSocket. SSE uses a standard HTTP connection that the server keeps open, sending events as plain text whenever new data is available. Because it is just regular HTTP, SSE works through every proxy, firewall, and CDN without any special configuration.

    The browser's `EventSource` API handles SSE natively, including automatic reconnection with the `Last-Event-ID` header so the client can resume exactly where it left off after a network interruption. This built-in reconnection logic is something you would have to build yourself with WebSocket, and getting it right (handling message ordering, deduplication, and state reconciliation) is surprisingly complex.

    SSE is ideal for live notifications, activity feeds, stock price tickers, build status updates, and any scenario where the client mostly listens. The client can still send data to the server using regular HTTP POST requests — it just cannot do so over the SSE connection itself. This combination — SSE for server-to-client events plus regular HTTP for client-to-server actions — covers a surprisingly large number of "real-time" use cases without the complexity of WebSocket.

    One practical consideration: SSE connections count against the browser's limit of six concurrent HTTP/1.1 connections per domain. If your application opens an SSE connection, that leaves only five connections for other requests. Under HTTP/2 this is not an issue because everything multiplexes over a single connection.

    **GitHub** uses SSE for their Actions workflow status page. When you watch a CI pipeline run, the build log streams to your browser over SSE. This is a natural fit: the server has data to push (log lines), the client just watches, and SSE's auto-reconnection ensures you never miss output even if your connection briefly drops. **ChatGPT** and other LLM-powered applications also use SSE to stream generated tokens to the browser in real time — each token arrives as an SSE event, giving the user that characteristic "typing" effect without waiting for the full response to generate.

    ### Long Polling (The Fallback)

    Long polling is the oldest real-time technique and the most universally compatible. The client sends a regular HTTP request, and the server holds it open without responding until new data is available (or a timeout expires, typically 30-60 seconds). When the client gets a response, it immediately sends another request, creating a loop that approximates real-time delivery.

    Long polling works everywhere — ancient browsers, corporate proxy servers behind strict packet inspectors, the most restrictive firewalls. It requires no special protocol support whatsoever. The cost is resource consumption: each connected client holds one HTTP connection open on the server. At scale, this means tens of thousands of open connections sitting idle most of the time, consuming memory, file descriptors, and load balancer capacity. There is also a latency penalty: when the timeout expires and the client reconnects, any event that arrives during the reconnection window is delayed until the next poll completes.

    ### Choosing the Right Approach: A Scenario Walkthrough

    Suppose you are building a project management tool. The feature list includes live notifications (someone assigned you a task), a real-time activity feed, and collaborative document editing. How do you choose?

    For notifications and the activity feed, data flows in one direction: server to client. SSE is the simplest fit. It works through corporate proxies (important for an enterprise tool), reconnects automatically, and the client can acknowledge notifications with a regular HTTP POST.

    For collaborative editing, both users need to send and receive changes simultaneously with minimal latency. WebSocket is the right choice here — Google Docs, Notion, and Figma all use WebSocket for their real-time collaboration features. You would implement long polling as a fallback for the small percentage of users behind problematic firewalls.

    The key insight is that you do not need to pick one protocol for your entire application. Different features have different requirements, and mixing protocols is both normal and expected.

    ### Comparison

    | | WebSocket | SSE | Long Polling |
    |---|---|---|---|
    | **Direction** | Bidirectional | Server to client | Server to client |
    | **Latency** | Sub-millisecond | Low (HTTP-level) | Medium (poll cycle) |
    | **Per-message overhead** | 2-10 bytes | ~50 bytes (text) | 500+ bytes (full HTTP) |
    | **Reconnection** | Manual (build it yourself) | Automatic (built-in) | Automatic (by design) |
    | **Proxy/firewall support** | Problematic | Excellent | Universal |
    | **Server resource cost** | Persistent connection | Persistent connection | Persistent connection |
    | **Best for** | Chat, gaming, collaboration | Notifications, feeds, tickers | Legacy compatibility |

    **Slack** uses WebSocket for real-time messaging across 10 million or more concurrent connections. When WebSocket is blocked by a user's network environment, Slack gracefully falls back to long polling to maintain connectivity — the user experience degrades slightly in latency but functionality never breaks entirely. **Notion** uses a similar approach for collaborative editing, with WebSocket as the primary channel and automatic fallback for restricted networks. This pattern of "prefer the fast protocol, fall back to the compatible one" appears in almost every production real-time system.

=== "Protocol Selection"

    Choosing between REST, gRPC, and other API protocols is one of the most consequential architectural decisions in a system design. This section covers the key trade-offs at a high level — for a deeper comparison including GraphQL, see the dedicated API Design guide.

    ### REST vs gRPC

    REST sends JSON over HTTP. The payload is human-readable, every programming language has an HTTP library, and browsers can call REST APIs directly with `fetch()`. This universality makes REST the default choice for public APIs and frontend-backend communication. The downside is performance: JSON is text-based and must be parsed character by character, HTTP headers add overhead on every request, and there is no built-in schema enforcement — clients and servers can silently disagree about the shape of data until something breaks at runtime.

    gRPC sends Protocol Buffers (a compact binary format) over HTTP/2. Serialization is 5-10x faster than JSON, payloads are 3-5x smaller, and the `.proto` schema file generates strongly-typed client and server code in dozens of languages automatically. When a service A calls service B calls service C, the cumulative savings from binary serialization and smaller payloads are substantial — a chain of five microservice calls that takes 50ms with REST/JSON might take 15-20ms with gRPC.

    Streaming is a first-class feature — gRPC supports server streaming, client streaming, and bidirectional streaming out of the box. The downsides are browser support (browsers cannot call gRPC directly without a gRPC-Web proxy layer) and debuggability (binary payloads require tooling like `grpcurl` or Postman's gRPC support to inspect, unlike JSON which you can read in any text editor or browser devtools).

    Use REST for public APIs, browser-facing services, and anywhere human readability and broad compatibility matter. Use gRPC for internal microservice communication, high-throughput data pipelines, and polyglot environments where schema enforcement prevents integration bugs across teams. In practice, the decision often comes down to who consumes the API: if it is external developers or browsers, choose REST; if it is your own services in a controlled environment, gRPC's performance and safety wins justify the tooling complexity.

    **Google** uses gRPC for virtually all internal service-to-service communication, handling billions of RPCs per second across their global infrastructure. Public-facing Google APIs (Maps, Cloud, YouTube Data API) expose REST endpoints because that is what external developers expect and what browsers can consume directly. **Netflix** similarly adopted gRPC for inter-service calls, citing the combination of strong typing and code generation as a major reduction in integration bugs across their hundreds of microservices.

    ### The Hybrid Approach

    Most production systems use multiple protocols simultaneously. A typical architecture might look like this:

    ```
    Mobile/Web Client
        |
        |── REST (JSON/HTTPS) ──> API Gateway ──> Backend Services
        |                                              |
        |── WebSocket ──────────> Notification         |── gRPC ──> User Service
        |                         Service              |── gRPC ──> Order Service
        |── SSE ────────────────> Activity             |── gRPC ──> Payment Service
                                  Feed Service
    ```

    The external boundary speaks REST and WebSocket because browsers understand them natively. Internal services communicate over gRPC for performance and type safety. Real-time features use WebSocket or SSE depending on whether they need bidirectional communication. This layered approach lets each protocol play to its strengths.

    ### Performance at a Glance

    | Protocol | Typical Payload Size | Relative Latency | Browser Support | Best For |
    |---|---|---|---|---|
    | **REST (JSON/HTTP)** | 1x (baseline) | 1x (baseline) | Native | Public APIs, frontend |
    | **gRPC (Protobuf/HTTP2)** | 0.2-0.3x | 0.3-0.5x | Via proxy only | Internal services |
    | **WebSocket** | Minimal framing | Sub-ms after connect | Native | Real-time bidirectional |
    | **SSE** | HTTP text stream | Low | Native | Server push |
    | **GraphQL** | Variable | 1-1.5x | Native | Flexible client queries |

    For an in-depth comparison of REST, GraphQL, and gRPC — including when to combine them in the same system — see [API Design](../communication/api-design/index.md).

---

## Common Interview Scenarios

When a system design interview involves protocol choices, the reasoning matters more than the answer. Here are the patterns interviewers look for:

**"Design a chat system"** -- WebSocket for message delivery (bidirectional, low latency), REST for message history and user management (stateless, cacheable), gRPC between backend services if microservices architecture.

**"Design a live sports scoreboard"** -- SSE from server to millions of clients (unidirectional push, works through CDNs), REST for fetching historical scores. WebSocket would be overkill since clients never send data.

**"Design an internal microservices platform"** -- gRPC for all service-to-service communication (type safety, performance, streaming), REST for any developer-facing API or admin dashboard.

**"Design a stock trading platform"** -- UDP for market data feeds (millions of price updates per second, stale data is worse than missing data), TCP/REST for order placement (every order must be reliably delivered), WebSocket for the trader's dashboard (bidirectional: show live prices and accept trades).

The key principle: match the protocol to the communication pattern, not to the system as a whole. A single system will almost always use multiple protocols for different purposes.

---

## Key Takeaways

1. **HTTP/2 is the right default for most APIs.** It gives you multiplexing and header compression with zero application-level changes over HTTP/1.1.

2. **Choose TCP when data integrity matters, UDP when freshness matters.** APIs, databases, and financial transactions need TCP. Video, voice, and gaming benefit from UDP's lower latency.

3. **WebSocket is for bidirectional real-time; SSE is for server-push.** If the client only needs to listen, SSE is simpler and far more infrastructure-friendly.

4. **Use gRPC between internal services, REST for external APIs.** gRPC's performance advantages compound across chains of microservice calls. REST's compatibility is unbeatable for public consumption.

5. **HTTP/3 matters most on mobile and unreliable networks.** Connection migration and 0-RTT resumption solve real problems that HTTP/2 cannot address.

6. **Always plan for fallback.** WebSocket needs a long-polling fallback. UDP-based protocols need TCP fallback. Corporate networks block more than you expect.

7. **Real systems use multiple protocols.** REST for public APIs, gRPC for internal services, WebSocket or SSE for real-time features. Match the protocol to the communication pattern, not the system.

---

## Related Topics

- **[Load Balancers](load-balancers.md)** -- distributing traffic across servers at L4 and L7
- **[Proxies](proxies.md)** -- reverse proxies, forward proxies, and protocol termination
- **[CDN](cdn.md)** -- edge delivery leveraging HTTP/2 and HTTP/3
- **[API Design](../communication/api-design/index.md)** -- in-depth REST vs GraphQL vs gRPC comparison
