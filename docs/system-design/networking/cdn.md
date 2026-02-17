# Content Delivery Networks (CDN)

When a user in Tokyo loads your website hosted in Virginia, every request travels 14,000 kilometers round trip. At the speed of light through fiber, that's roughly 70ms of pure physics — and real-world latency is 2-3x worse due to routing, congestion, and processing. Multiply by dozens of assets (HTML, CSS, JavaScript, images) and the page takes seconds to load.

A CDN solves this by placing copies of your content on servers around the world, so the user in Tokyo gets served from a server in Tokyo. The difference between 200ms and 20ms latency is the difference between a site that feels instant and one that feels sluggish.

This guide explains how CDNs work, what decisions you need to make when using one, and how companies at different scales approach content delivery.

---

## How a CDN Works

A CDN is a network of geographically distributed servers (called **edge servers** or **Points of Presence / PoPs**) that cache and serve content close to users. The basic flow:

```
First request (cache miss):

User in Tokyo ──→ Edge Server (Tokyo) ──→ Origin Server (Virginia)
                         │                        │
                    "I don't have             "Here's the file"
                     this file"                    │
                         │◄────────────────────────┘
                    Stores a copy,
                    serves to user

Subsequent requests (cache hit):

User in Tokyo ──→ Edge Server (Tokyo)
                         │
                    "I have this file!"
                    Serves directly (5-20ms)

User in London ──→ Edge Server (London) ──→ Origin Server (Virginia)
                         │                     (only on first request)
                    Same process for
                    each PoP independently
```

The CDN sits between users and your origin server (where the content actually lives). It intercepts requests, serves cached content when possible, and fetches from origin only when needed.

---

=== "What to Cache"

    Not all content benefits equally from CDN caching. The value depends on how cacheable the content is and how frequently it's requested.

    ### Static Assets (High Value)

    Images, CSS files, JavaScript bundles, fonts, and videos are the sweet spot for CDNs. They don't change between users, they're requested frequently, and they're often the largest files on a page.

    A typical web page loads 50-100 static assets. Serving these from a CDN edge 20ms away instead of an origin 200ms away saves 9-18 *seconds* of cumulative latency (accounting for connection setup and sequential loading).

    **Netflix** serves all video content through their Open Connect CDN — purpose-built appliances placed directly in ISP networks. A movie streamed in Sao Paulo never leaves the ISP's data center.

    ### Semi-Dynamic Content (Medium Value)

    API responses that are the same for many users — product listings, news feeds, search results for popular queries. These can be cached with short TTLs (30 seconds to 5 minutes). The content might be slightly stale, but the latency reduction is worth it.

    **Twitter** caches timeline API responses at their edge. A user's timeline doesn't change every second, so serving a 30-second-old version from cache dramatically reduces origin load.

    ### Personalized Content (Low CDN Value)

    Content unique to each user — their inbox, their cart, their dashboard. Traditional CDN caching doesn't help because every user gets different content. However, modern CDN edge computing (Cloudflare Workers, Lambda@Edge) can assemble personalized content at the edge by combining cached templates with user-specific data.

=== "Cache Invalidation"

    The hardest problem in CDN management. When you deploy a new version of your website or update a product price, how do you make sure users see the new version instead of stale cached content?

    ### Versioned URLs (Cache Busting)

    The most reliable approach: include a version identifier in the URL. When the content changes, the URL changes, so the CDN treats it as a new resource.

    ```
    Old: /static/app.js          → cached for 1 year
    New: /static/app.v2.3.1.js   → new URL, new cache entry

    Or using content hashes:
    /static/app.a1b2c3d4.js      → hash of file contents in the filename
    ```

    This lets you set very long cache TTLs (1 year) for static assets because the URL only changes when the content changes. Webpack, Vite, and other build tools generate these hashed filenames automatically.

    ### TTL-Based Expiry

    Set a Time-To-Live on cached content. After the TTL expires, the edge server fetches a fresh copy from origin on the next request.

    ```
    Cache-Control: max-age=3600    → cache for 1 hour
    Cache-Control: max-age=86400   → cache for 1 day
    Cache-Control: no-cache         → always revalidate with origin
    Cache-Control: no-store         → never cache
    ```

    Typical TTL strategy:
    - **Static assets with versioned URLs:** 1 year (365 days)
    - **Images and media:** 1 day to 1 week
    - **HTML pages:** 5 minutes to 1 hour
    - **API responses:** 30 seconds to 5 minutes
    - **User-specific content:** no-store (don't cache)

    ### Purge / Invalidation API

    Force the CDN to drop cached content immediately. Every CDN provider offers an API for this:

    - **Cloudflare:** purge by URL, prefix, tag, or everything
    - **CloudFront:** create an invalidation for specific paths (takes 5-10 minutes to propagate)
    - **Fastly:** instant purge (sub-second propagation) — this is Fastly's key differentiator

    Purging is useful for emergency content removal or critical updates, but shouldn't be your primary invalidation strategy. Versioned URLs and TTLs are more reliable at scale.

=== "Architecture"

    ### Origin Shield

    Without an origin shield, every edge server independently fetches from your origin when it has a cache miss. If you have 200 PoPs and deploy a new version, your origin potentially handles 200 simultaneous requests for each asset.

    An **origin shield** is an intermediate caching layer between edge servers and the origin. Edge servers fetch from the shield instead of directly from origin. The shield handles origin fetching, so your origin only sees one request per asset regardless of how many edge servers need it.

    ```
    Without origin shield:              With origin shield:

    Edge (Tokyo) ──→ Origin             Edge (Tokyo) ──→ Shield ──→ Origin
    Edge (London) ──→ Origin            Edge (London) ──→ Shield
    Edge (Sydney) ──→ Origin            Edge (Sydney) ──→ Shield
    Edge (Mumbai) ──→ Origin            Edge (Mumbai) ──→ Shield
       4 origin requests                   1 origin request
    ```

    This dramatically reduces origin load during cache misses, especially after deployments or cache purges. Most CDN providers offer this as a configuration option.

    ### Push vs Pull

    **Pull CDN (most common):** Edge servers fetch content from origin on demand, on the first request. Simple to set up — just point the CDN at your origin. The downside: the first user to request each asset from each edge location experiences higher latency (cache miss penalty).

    **Push CDN:** You explicitly upload content to the CDN ahead of time. Useful for large media files (video), where you want content pre-positioned before users request it. More complex to manage but eliminates cold-cache latency. **YouTube** pre-positions popular videos in ISP-level caches.

    ### Multi-CDN

    Large companies use multiple CDN providers simultaneously for redundancy and performance:

    - **Failover:** If one CDN has an outage, traffic automatically shifts to another
    - **Performance optimization:** Route each user to whichever CDN is fastest for their location
    - **Cost optimization:** Different CDNs have different pricing for different regions
    - **Vendor leverage:** Avoid lock-in to a single provider

    **Apple** uses multiple CDNs (including their own infrastructure) for App Store downloads and software updates. **GitHub** uses Fastly as their primary CDN but has failover to other providers.

---

## CDN Providers Compared

| Provider | Edge Locations | Strength | Best For |
|---|---|---|---|
| **Cloudflare** | 300+ cities | Security (DDoS, WAF), free tier, edge computing (Workers) | Most web applications, security-focused |
| **AWS CloudFront** | 400+ PoPs | Deep AWS integration, Lambda@Edge | AWS-native applications |
| **Fastly** | 60+ PoPs | Instant purge (<150ms), VCL programmability, real-time logging | Dynamic content, media companies |
| **Akamai** | 4,100+ PoPs | Largest network, enterprise features | Largest enterprises, media streaming |
| **Google Cloud CDN** | 140+ PoPs | Google network backbone, Cloud Armor integration | GCP-native applications |

For most applications, **Cloudflare** is the best starting point — generous free tier, excellent performance, built-in security, and edge computing with Workers. Move to a specialized provider when you have specific needs (Fastly for instant purge, Akamai for massive scale, CloudFront for AWS integration).

---

## Performance Optimization at the Edge

Beyond caching, modern CDNs optimize content delivery in several ways:

**Compression.** CDNs automatically compress text-based responses (HTML, CSS, JavaScript, JSON) using Gzip or Brotli. Brotli typically achieves 15-20% better compression than Gzip, reducing file sizes by 70-80%.

**Image optimization.** Automatically convert images to modern formats (WebP, AVIF) based on what the user's browser supports. Resize images to match the device's screen size. A 4000x3000 photo doesn't need to be sent at full resolution to a mobile phone with a 390px-wide viewport.

**HTTP/2 and HTTP/3.** Modern protocols that allow multiplexing (multiple requests over a single connection), header compression, and server push. CDN edge servers often support HTTP/3 (QUIC) even when your origin only speaks HTTP/1.1.

**Edge computing.** Run code at the CDN edge instead of at your origin. Cloudflare Workers, Lambda@Edge, and Fastly Compute handle tasks like A/B testing, authentication, geolocation-based routing, and personalization without a round trip to origin.

---

## Key Takeaways

1. **CDNs turn distance into a non-issue.** Serving content from 20ms away instead of 200ms away makes the biggest single improvement to perceived performance.

2. **Versioned URLs are the gold standard for cache invalidation.** Content-hashed filenames (app.a1b2c3.js) let you cache aggressively without worrying about stale content.

3. **Use origin shields to protect your origin server.** Without one, every edge PoP independently hammers your origin on cache misses.

4. **Static assets are table stakes — the real value is caching semi-dynamic content.** Cache API responses with short TTLs to reduce origin load by 10-100x.

5. **Cloudflare is the default starting point** for most applications. Move to specialized providers when specific needs arise (instant purge, AWS integration, massive scale).

6. **Edge computing is the future of CDNs.** Running logic at the edge (auth, personalization, A/B testing) eliminates origin round trips entirely.

---

## Related Topics

- **[Load Balancers](load-balancers.md)** — distributing traffic across servers
- **[Proxies](proxies.md)** — CDNs are specialized reverse proxies
- **[Database Scaling Patterns](../data/databases/scaling-patterns.md)** — caching as a database scaling strategy
