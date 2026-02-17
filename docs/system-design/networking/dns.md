# DNS (Domain Name System)

Every time you type a URL into a browser, something has to translate `api.example.com` into `203.0.113.42` before a single byte of data can flow. That something is DNS -- the internet's phone book. It's the most heavily queried distributed system on the planet, silently resolving billions of domain names every hour, and yet most software engineers only think about it when something breaks.

Understanding DNS matters beyond trivia. It affects how you deploy services, how fast your application feels to users around the world, how you handle failover between regions, and how you route traffic during blue-green deployments. When DNS goes wrong, everything goes wrong -- the 2021 Facebook outage that took down Facebook, Instagram, and WhatsApp for six hours was a DNS misconfiguration.

---

## DNS Record Types at a Glance

| Record | Purpose | Example | When You'll Use It |
|--------|---------|---------|-------------------|
| **A** | Maps domain to IPv4 address | `example.com -> 93.184.216.34` | Pointing domain to a server |
| **AAAA** | Maps domain to IPv6 address | `example.com -> 2606:2800:220:1:...` | IPv6-enabled infrastructure |
| **CNAME** | Alias to another domain | `www.example.com -> example.com` | Subdomains, CDN setup |
| **MX** | Mail server routing | `example.com -> mail.example.com` | Email delivery |
| **TXT** | Arbitrary text data | `v=spf1 include:_spf.google.com` | Domain verification, email auth |
| **SRV** | Service discovery (host + port) | `_http._tcp.example.com -> 8080` | Kubernetes, SIP, XMPP |
| **NS** | Delegates to nameserver | `example.com -> ns1.cloudflare.com` | Changing DNS providers |

---

=== "How DNS Works"

    When you type `www.netflix.com` into your browser and press Enter, the browser needs an IP address before it can open a TCP connection. What follows is a chain of lookups, each one narrowing the search until the exact address is found. Most of the time, a cache short-circuits the chain early -- but understanding the full path matters for troubleshooting and system design.

    ### The Resolution Chain

    ```
    You type: www.netflix.com

    Step 1: Browser Cache        (~0ms)
    Browser checks its own DNS cache (Chrome: chrome://net-internals/#dns)
    Cache HIT? Done. Cache MISS? Continue.
            |
    Step 2: OS Cache             (~0ms)
    Operating system checks /etc/hosts and its resolver cache
    Cache HIT? Done. Cache MISS? Continue.
            |
    Step 3: Recursive Resolver   (~1-5ms)
    Your ISP's resolver (or 1.1.1.1 / 8.8.8.8) checks its cache
    Cache HIT? Done. Cache MISS? It starts the recursive lookup:
            |
    Step 4: Root Nameserver      (~2-10ms)
    "I don't know netflix.com, but .com is handled by these TLD servers"
    13 root server clusters (a.root-servers.net through m.root-servers.net)
    Distributed across 1,500+ anycast instances worldwide
            |
    Step 5: TLD Nameserver       (~2-10ms)
    ".com TLD here. netflix.com is delegated to these authoritative servers"
    Verisign operates the .com TLD servers, handling 30B+ queries/day
            |
    Step 6: Authoritative NS     (~2-10ms)
    "www.netflix.com? That's a CNAME to dualstack.apiproxy-website-us-east-1...
     which resolves to 54.237.226.164"
    This is the definitive answer -- the server the domain owner configured
            |
    Response flows back through the chain. Each layer caches the result.
    Total time (cold cache): 20-120ms
    Total time (warm cache): <1ms
    ```

    The recursive resolver does the heavy lifting. It walks the DNS tree from root to authoritative, assembling the answer step by step. Your browser and OS just ask the resolver and wait.

    ### Caching at Every Layer

    DNS would collapse under its own weight without aggressive caching. Every response includes a TTL (Time To Live) that tells the recipient how long to cache the answer.

    The browser caches DNS results for roughly 60 seconds (Chrome) to several minutes, regardless of the record's TTL -- browsers enforce a minimum to avoid hammering resolvers. The operating system maintains its own cache, typically honoring the record's TTL. The recursive resolver also caches, and this is where the biggest wins happen: a popular domain like `google.com` with a 300-second TTL will be cached by every major resolver, meaning the vast majority of lookups for that domain never leave the resolver.

    In practice, when Cloudflare's 1.1.1.1 resolver handles over 500 billion queries per day, the majority are served from cache. The resolver only contacts authoritative nameservers for cache misses -- roughly 10-15% of total queries.

    ### What "Resolving" Actually Means for Your Application

    From your application's perspective, a DNS lookup is a blocking operation that happens before any HTTP request. If DNS resolution takes 100ms and your API call takes 50ms, DNS is the bottleneck. This is why connection pooling and keep-alive connections matter -- they reuse existing TCP connections, skipping DNS entirely for subsequent requests to the same host.

=== "Record Types"

    Each DNS record type serves a specific purpose, and knowing which to use in different scenarios is practical knowledge you'll need when setting up infrastructure.

    ### A and AAAA Records

    The most fundamental record types. An A record maps a domain name to an IPv4 address, and an AAAA record maps it to an IPv6 address. You can have multiple A records for the same domain -- the resolver returns all of them, and the client picks one (usually the first, or randomly).

    ```
    example.com.    300    IN    A       93.184.216.34
    example.com.    300    IN    AAAA    2606:2800:220:1:248:1893:25c8:1946
    ```

    When you configure a server in AWS and point your domain to its Elastic IP, you're creating an A record. Most domains today should have both A and AAAA records -- IPv6 adoption has crossed 40% globally and is growing.

    ### CNAME Records (and the Apex Domain Trap)

    A CNAME (Canonical Name) record creates an alias -- it says "this domain is really just another name for that domain." When a resolver encounters a CNAME, it follows the chain until it reaches an A or AAAA record.

    ```
    www.example.com.    300    IN    CNAME    example.com.
    blog.example.com.   300    IN    CNAME    mysite.netlify.app.
    ```

    This is how you point a custom domain to GitHub Pages or a CDN: you create a CNAME from your subdomain to the provider's domain, and they handle the IP addresses on their end. If they change IPs, your domain follows automatically.

    The critical limitation: **you cannot place a CNAME at the apex (naked) domain.** The DNS specification (RFC 1034) forbids a CNAME from coexisting with other record types at the same name, and the apex domain must have NS and SOA records. So `www.example.com CNAME cdn.provider.com` works, but `example.com CNAME cdn.provider.com` does not. This is why Cloudflare invented CNAME flattening (covered in Common Problems).

    ### MX Records

    MX (Mail Exchange) records tell the world which servers handle email for your domain. Each MX record includes a priority value -- lower numbers are tried first.

    ```
    example.com.    300    IN    MX    10    mail1.example.com.
    example.com.    300    IN    MX    20    mail2.example.com.
    ```

    When someone sends email to `user@example.com`, their mail server looks up the MX records for `example.com`, then connects to the server with the lowest priority number. If that server is down, it tries the next one. If you use Google Workspace, you point your MX records to Google's mail servers; if you use Microsoft 365, you point them to Microsoft's.

    ### TXT Records

    TXT records hold arbitrary text and have become the Swiss Army knife of DNS. Their most common uses are domain verification and email authentication.

    When AWS Certificate Manager asks you to "add a CNAME record to verify domain ownership," or when Google asks you to "add a TXT record with this verification string," they're using DNS as a proof-of-ownership mechanism. Only the domain owner can add records, so the ability to add a specific record proves ownership.

    For email, TXT records carry SPF (which servers can send email for your domain), DKIM (cryptographic signatures on emails), and DMARC (what to do with emails that fail SPF/DKIM checks). These are critical for preventing email spoofing and ensuring your emails don't land in spam.

    ### SRV Records

    SRV records are the only standard DNS record type that includes a port number, making them useful for service discovery. The format includes priority, weight, port, and target.

    ```
    _http._tcp.example.com.  300  IN  SRV  10 60 8080 server1.example.com.
    _http._tcp.example.com.  300  IN  SRV  10 40 8080 server2.example.com.
    ```

    Kubernetes uses SRV records internally for service discovery -- when one pod needs to find another service, the cluster DNS returns SRV records with the correct port and host. Outside of Kubernetes, SRV records are used by SIP (VoIP), XMPP (chat), and LDAP.

    ### Practical Setup: GitHub Pages Custom Domain

    A concrete example tying several record types together. To serve `www.mysite.com` from GitHub Pages, you create a CNAME record pointing `www.mysite.com` to `myusername.github.io`. For the apex domain `mysite.com`, you create four A records pointing to GitHub's IP addresses. GitHub verifies ownership through a TXT record or a CNAME file in your repository.

=== "DNS as Load Balancing"

    DNS can do more than simple name-to-address translation. By returning different IP addresses based on various criteria, DNS becomes a coarse-grained load balancing layer -- the first point of traffic distribution before requests even reach your infrastructure.

    ### Round-Robin DNS

    The simplest form of DNS load balancing: configure multiple A records for the same domain, and the resolver returns them all. Clients typically connect to the first address in the list, and most resolvers rotate the order with each query.

    ```
    api.example.com.    60    IN    A    10.0.1.1
    api.example.com.    60    IN    A    10.0.1.2
    api.example.com.    60    IN    A    10.0.1.3
    ```

    This distributes traffic roughly evenly across servers without any load balancer hardware. But "roughly" is doing a lot of work in that sentence -- DNS round robin has no awareness of server health, current load, or connection counts. A server that's on fire still receives its share of traffic until someone manually removes its DNS record (and then waits for caches to expire).

    ### AWS Route 53 Routing Policies

    Route 53 turns DNS into a sophisticated traffic management layer. Each routing policy uses health checks to avoid sending traffic to unhealthy endpoints, and the routing decision happens at the DNS level -- before the client opens a connection.

    | Policy | Use When | Example | Trade-off |
    |--------|----------|---------|-----------|
    | **Simple** | One resource, no health checks | Single-region API | No failover, no load distribution |
    | **Weighted** | Gradual rollouts, A/B testing | 90% to v1, 10% to v2 | Caching skews actual percentages |
    | **Latency** | Multi-region, minimize response time | US users to us-east-1, EU users to eu-west-1 | Latency measurements lag reality |
    | **Failover** | Active-passive disaster recovery | Primary in us-east-1, DR in us-west-2 | Failover speed limited by TTL |
    | **Geolocation** | Regulatory compliance, localized content | EU users always routed to EU servers | Users in unmapped regions need a default |
    | **Multivalue** | Simple load distribution with health checks | Up to 8 healthy IPs returned per query | No weighting, no latency optimization |

    Weighted routing is particularly useful during deployments. Set the new deployment to weight 10 and the old one to weight 90 -- 10% of DNS resolutions will point to the new version. If something goes wrong, set the new deployment's weight to 0 and traffic drains as DNS caches expire.

    ### Limitations of DNS Load Balancing

    DNS load balancing is powerful for the first layer of traffic distribution, but it has fundamental limitations that make it unsuitable as your only balancing strategy.

    **Slow failover.** When a server goes down, DNS-based load balancing can only react as fast as the TTL allows. With a 300-second TTL, some clients will keep hitting the dead server for up to 5 minutes after failure. Even a 60-second TTL means a full minute of errors for some users. Compare this to a hardware load balancer that detects failure in seconds and re-routes instantly.

    **No session affinity.** DNS has no concept of sessions. A user might resolve the domain to Server A, then on their next request (after the browser's DNS cache expires), resolve to Server B. If your servers maintain local state, this breaks the user's session.

    **Caching distortion.** You set a weight of 10/90 in Route 53, but the actual traffic split might be 5/95 or 15/85 because resolvers cache the answer and serve it to thousands of users. One resolver caching the "10% answer" and serving a large corporate network can drastically skew your distribution.

    Netflix uses Route 53 latency-based routing to direct users to the nearest AWS region -- but once traffic arrives at the region, an actual load balancer (not DNS) handles distribution across servers. DNS picks the region; load balancers pick the server.

=== "Common Problems"

    DNS problems are uniquely frustrating because they're invisible, delayed, and affect everything. Understanding the common failure modes saves hours of debugging.

    ### Propagation Delays

    "I updated the DNS record 5 minutes ago. Why doesn't it work?" This is the most common DNS complaint, and it stems from a misunderstanding of how caching works. When you change a DNS record, the old value is still cached at resolvers worldwide. Those caches don't expire until their TTL runs out -- and there's no way to force them to refresh.

    If your old record had a TTL of 86400 (24 hours), some users will see the old IP for up to a full day after you make the change. This is why GitHub's engineering team, during their 2018 DNS migration, lowered their TTLs to 60 seconds a full 24 hours before the actual migration. This ensured that by the time the real change happened, no resolver in the world had a cache entry older than 60 seconds.

    **TTL strategy by scenario:**

    | Scenario | TTL | Why |
    |----------|-----|-----|
    | Preparing for a migration | 60s | Minimize stale cache window |
    | Blue-green deployment | 60s | Fast traffic switching |
    | Normal operations | 300s (5 min) | Balance between freshness and resolver load |
    | Stable infrastructure (rarely changes) | 86400s (24h) | Reduce resolver queries, faster client lookups |
    | During an active incident | Lower to 60s (after current TTL expires) | Prepare for rapid changes |

    ### CNAME Flattening

    The apex domain restriction on CNAMEs is a real operational headache. You want `example.com` (no www) to point to your CDN or load balancer, but the CDN gives you a hostname (`d1234.cloudfront.net`), not an IP address. You can't use a CNAME at the apex, and you don't want to hardcode an IP in an A record because the CDN's IP might change.

    Cloudflare's solution -- CNAME flattening -- resolves the CNAME chain at the authoritative nameserver level and returns the resulting A/AAAA records directly. From the querying resolver's perspective, it looks like a normal A record response, but behind the scenes, Cloudflare is following the CNAME chain on your behalf. AWS Route 53 offers a similar feature called "alias records," and other providers have adopted similar approaches.

    ### Split-Horizon DNS

    In many organizations, the same domain name needs to resolve to different addresses depending on who's asking. `api.company.com` might resolve to a public IP (203.0.113.10) for external users but to a private IP (10.0.1.50) for employees on the corporate network.

    This is split-horizon DNS (also called split-brain DNS). The authoritative nameserver returns different answers based on the source IP of the query. It's commonly used to route internal traffic directly to backend servers (skipping the external load balancer), to provide access to internal-only services, or to serve different content to different networks.

    ### DNS-over-HTTPS (DoH) and DNS-over-TLS (DoT)

    Traditional DNS queries are sent in plaintext over UDP port 53. Anyone on the network path -- your ISP, a coffee shop's WiFi operator, a government -- can see every domain you visit, even if the site itself uses HTTPS. The URL path and content are encrypted, but the DNS lookup that happens first is completely exposed.

    DNS-over-HTTPS (DoH) wraps DNS queries inside HTTPS, encrypting them and making them indistinguishable from normal web traffic. DNS-over-TLS (DoT) encrypts queries using TLS on a dedicated port (853). Both prevent eavesdropping on DNS queries. Firefox and Chrome support DoH natively, and Cloudflare's 1.1.1.1 and Google's 8.8.8.8 both support both protocols.

    The trade-off is centralization: when every browser sends DNS queries to Cloudflare or Google over HTTPS, those companies gain visibility into global browsing patterns. This shifts trust from ISPs to DNS providers -- whether that's better depends on your threat model.

    ### Troubleshooting with dig and nslookup

    When DNS isn't behaving as expected, `dig` is the essential diagnostic tool. It shows exactly what a resolver returns, including TTLs, record types, and the full resolution chain.

    ```bash
    # Basic lookup -- what does the resolver return?
    dig example.com

    # Query a specific nameserver directly (bypass cache)
    dig @8.8.8.8 example.com

    # Trace the full resolution path (root -> TLD -> authoritative)
    dig +trace example.com

    # Check a specific record type
    dig example.com MX
    dig example.com TXT
    ```

    The `+trace` flag is particularly valuable -- it shows every step of the resolution chain, making it clear exactly where a problem occurs. If the authoritative server returns the correct answer but your resolver doesn't, you have a caching issue. If the authoritative server returns the wrong answer, you have a configuration issue.

---

## DNS Providers Compared

| Provider | PoPs / Anycast Locations | Query Latency | Key Feature | Best For |
|----------|--------------------------|---------------|-------------|----------|
| **Cloudflare** | 300+ cities | ~11ms global avg | Free tier, CNAME flattening, DDoS protection | Most applications, apex domain CNAME needs |
| **AWS Route 53** | 80+ edge locations | ~20-40ms | Alias records, deep AWS integration, routing policies | AWS-native infrastructure |
| **Google Cloud DNS** | Google's global network | ~15-25ms | 100% SLA, tight GCP integration | GCP-native infrastructure |
| **NS1** | 25+ PoPs | ~10-15ms | Filter chains, real-time traffic steering, API-first | Advanced traffic management, edge cases |

For most teams, the choice follows your cloud provider: Route 53 if you're on AWS, Cloud DNS if you're on GCP. If you need advanced features like CNAME flattening at the apex domain or a strong free tier, Cloudflare is the default. NS1 is the specialist choice when you need programmable DNS with complex routing logic.

---

## Key Takeaways

1. **DNS is a distributed caching system, not a real-time lookup.** Every layer caches aggressively. When you change a record, the old value persists until TTLs expire worldwide. Plan accordingly.

2. **Lower your TTLs before any migration.** Drop TTLs to 60 seconds at least 24 hours in advance. This is the single most important operational DNS practice.

3. **DNS load balancing is a coarse first layer, not a replacement for real load balancers.** Use DNS to route to the right region; use load balancers to route to the right server.

4. **You cannot CNAME the apex domain.** Use CNAME flattening (Cloudflare), alias records (Route 53), or ANAME records if your provider supports them.

5. **TXT records are your proof-of-ownership mechanism.** Every cloud service uses them for domain verification. Expect to create TXT records when setting up SSL certificates, email, and SaaS integrations.

6. **When debugging, use `dig +trace` to walk the resolution chain.** It shows exactly where the problem is -- caching, delegation, or misconfiguration at the authoritative server.

---

## Related Topics

- **[Load Balancers](load-balancers.md)** -- DNS routes to regions; load balancers route to servers
- **[CDN](cdn.md)** -- CDNs rely on DNS (often anycast) to route users to the nearest edge
- **[Proxies](proxies.md)** -- reverse proxies and edge proxies that sit behind DNS resolution
