# System Design Problems

Master system design interviews with 50 real-world problems asked by top tech companies. Each problem follows the 4-step framework with architectural diagrams, capacity planning, and optimization strategies.

**Status:** ✅ 1 Complete | 🚧 0 In Progress | 📋 49 Planned

---

## Quick Start

| Your Goal | Start Here | Time Needed |
|-----------|------------|-------------|
| 🎯 **Interview next week** | Browse by Company → Pick top 10 | 7-10 days |
| 📚 **Learn system design** | Learning Path → Week 1 | 8 weeks |
| 🔍 **Practice specific concept** | Browse by Concept → Pick category | 1-2 weeks |
| ⚡ **Quick review** | Browse by Difficulty → Start Easy | 3-5 days |

---

=== "📁 By Category"

    ## Storage & Content Systems

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | ⭐⭐⭐⭐⭐ | Amazon, Google, Meta, Microsoft, Uber | Short code generation, caching, analytics |
    | **Pastebin** | 🟡 Medium | ⭐⭐⭐⭐ | Amazon, Microsoft, Twitter | Text storage, expiration, syntax highlighting |
    | **File Upload Service** | 🟡 Medium | ⭐⭐⭐⭐ | Dropbox, Google, Microsoft | Chunking, resumable uploads, deduplication |
    | **Image Hosting** | 🟡 Medium | ⭐⭐⭐⭐ | Instagram, Pinterest, Imgur | Image processing, CDN, thumbnails |
    | **Cloud Storage** | 🔴 Hard | ⭐⭐⭐⭐ | Dropbox, Google Drive, OneDrive | Sync, conflict resolution, versioning |

    ## Social & Communication

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Twitter Feed** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Twitter, Meta, LinkedIn | Fan-out, timelines, real-time updates |
    | **Instagram** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Meta, Instagram, TikTok | Photo storage, feeds, followers graph |
    | **WhatsApp/Chat** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Meta, WhatsApp, Slack, Discord | Real-time messaging, presence, group chat |
    | **Notification System** | 🟡 Medium | ⭐⭐⭐⭐ | All companies | Push notifications, delivery guarantees |
    | **News Feed** | 🔴 Hard | ⭐⭐⭐⭐ | Facebook, LinkedIn, Reddit | Ranking, personalization, real-time |

    ## Media & Entertainment

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Video Streaming** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Netflix, YouTube, Amazon | Adaptive bitrate, CDN, encoding |
    | **Music Streaming** | 🔴 Hard | ⭐⭐⭐⭐ | Spotify, Apple Music | Audio delivery, playlists, recommendations |
    | **Live Streaming** | 🔴 Hard | ⭐⭐⭐⭐ | Twitch, YouTube Live | Low latency, chat, transcoding |
    | **Video Conferencing** | 🔴 Hard | ⭐⭐⭐⭐ | Zoom, Google Meet, Teams | WebRTC, signaling, mixing |

    ## Search & Discovery

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Search Engine** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Google, Bing | Crawling, indexing, ranking |
    | **Autocomplete/Typeahead** | 🟡 Medium | ⭐⭐⭐⭐⭐ | Google, Amazon, Netflix | Trie, caching, prefix matching |
    | **Recommendation System** | 🔴 Hard | ⭐⭐⭐⭐ | Netflix, Amazon, YouTube | Collaborative filtering, ML models |
    | **Web Crawler** | 🟡 Medium | ⭐⭐⭐⭐ | Google, Bing, Archive.org | Queue, deduplication, politeness |

    ## E-Commerce & Payments

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **E-Commerce Platform** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Amazon, eBay, Shopify | Inventory, orders, payments, catalog |
    | **Payment System** | 🔴 Hard | ⭐⭐⭐⭐ | Stripe, PayPal, Square | Transactions, idempotency, ledger |
    | **Ticket Booking** | 🔴 Hard | ⭐⭐⭐⭐ | BookMyShow, Ticketmaster | Concurrency, seat locking, inventory |
    | **Food Delivery** | 🔴 Hard | ⭐⭐⭐⭐ | UberEats, DoorDash, GrubHub | Matching, routing, real-time tracking |

    ## Location-Based Services

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Ride Sharing** | 🔴 Hard | ⭐⭐⭐⭐⭐ | Uber, Lyft | Geohashing, matching, ETA, surge pricing |
    | **Yelp/Nearby Places** | 🟡 Medium | ⭐⭐⭐⭐ | Yelp, Google Maps | Geospatial indexing, quadtree |
    | **Google Maps** | 🔴 Hard | ⭐⭐⭐⭐ | Google, Apple | Routing, traffic, graph algorithms |
    | **Location Tracking** | 🟡 Medium | ⭐⭐⭐ | Uber, DoorDash, Find My | GPS data, geofencing, privacy |

    ## Infrastructure & Developer Tools

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Rate Limiter** | 🟡 Medium | ⭐⭐⭐⭐⭐ | All companies | Token bucket, sliding window, distributed |
    | **API Gateway** | 🟡 Medium | ⭐⭐⭐⭐ | Kong, AWS, Google | Routing, authentication, rate limiting |
    | **Distributed Cache** | 🔴 Hard | ⭐⭐⭐⭐ | Redis, Memcached | Consistent hashing, replication, eviction |
    | **Message Queue** | 🔴 Hard | ⭐⭐⭐⭐ | Kafka, RabbitMQ, SQS | Partitioning, ordering, delivery guarantees |
    | **Load Balancer** | 🟡 Medium | ⭐⭐⭐⭐ | All companies | Algorithms, health checks, sticky sessions |

    ## Collaboration & Productivity

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Google Docs** | 🔴 Hard | ⭐⭐⭐⭐ | Google, Microsoft, Notion | CRDT, OT, real-time collaboration |
    | **Calendar System** | 🟡 Medium | ⭐⭐⭐ | Google, Microsoft, Apple | Availability, conflicts, recurring events |
    | **Task Management** | 🟡 Medium | ⭐⭐⭐ | Asana, Jira, Trello | Projects, workflows, notifications |
    | **Code Repository** | 🔴 Hard | ⭐⭐⭐ | GitHub, GitLab | Version control, merge, diff |

    ## Analytics & Monitoring

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | **Analytics Platform** | 🔴 Hard | ⭐⭐⭐⭐ | Google Analytics, Mixpanel | Event tracking, aggregation, dashboards |
    | **Metrics Monitoring** | 🟡 Medium | ⭐⭐⭐⭐ | Datadog, Prometheus | Time-series DB, alerting, visualization |
    | **Log Aggregation** | 🟡 Medium | ⭐⭐⭐ | Splunk, ELK Stack | Collection, indexing, search |
    | **Distributed Tracing** | 🔴 Hard | ⭐⭐⭐ | Jaeger, Zipkin | Trace IDs, spans, correlation |

=== "🎚️ By Difficulty"

    ## 🟢 Easy Problems (5 problems)

    **Perfect for:** Beginners, first interview prep, understanding basics

    **Time per problem:** 30-40 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | **Key-Value Store** | Infrastructure | ⭐⭐⭐ | Basic storage, CRUD operations |
    | **URL Validator** | Tools | ⭐⭐⭐ | API design, validation logic |
    | **Simple Cache** | Infrastructure | ⭐⭐⭐⭐ | LRU, eviction policies |
    | **Unique ID Generator** | Infrastructure | ⭐⭐⭐⭐ | Distributed ID generation |
    | **Health Checker** | Monitoring | ⭐⭐⭐ | Polling, alerting basics |

    ---

    ## 🟡 Medium Problems (25 problems)

    **Perfect for:** Intermediate prep, common interview questions, building fundamentals

    **Time per problem:** 45-60 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | [**URL Shortener**](url-shortener.md) | Storage | ⭐⭐⭐⭐⭐ | Code generation, caching, analytics |
    | **Pastebin** | Storage | ⭐⭐⭐⭐ | Text storage, expiration handling |
    | **Rate Limiter** | Infrastructure | ⭐⭐⭐⭐⭐ | Token bucket, sliding window |
    | **Autocomplete** | Search | ⭐⭐⭐⭐⭐ | Trie, prefix matching, caching |
    | **File Upload Service** | Storage | ⭐⭐⭐⭐ | Chunking, resumable uploads |
    | **Image Hosting** | Storage | ⭐⭐⭐⭐ | CDN, image processing |
    | **Notification System** | Social | ⭐⭐⭐⭐ | Fan-out, delivery guarantees |
    | **Web Crawler** | Search | ⭐⭐⭐⭐ | Queue, deduplication |
    | **Yelp/Nearby** | Location | ⭐⭐⭐⭐ | Geospatial indexing |
    | **Calendar System** | Collaboration | ⭐⭐⭐ | Conflicts, availability |
    | **Task Management** | Collaboration | ⭐⭐⭐ | Workflows, notifications |
    | **API Gateway** | Infrastructure | ⭐⭐⭐⭐ | Routing, auth, rate limiting |
    | **Load Balancer** | Infrastructure | ⭐⭐⭐⭐ | Algorithms, health checks |
    | **Metrics Monitoring** | Monitoring | ⭐⭐⭐⭐ | Time-series, aggregation |
    | **Log Aggregation** | Monitoring | ⭐⭐⭐ | Collection, indexing |
    | **Location Tracking** | Location | ⭐⭐⭐ | GPS, geofencing |

    ---

    ## 🔴 Hard Problems (20 problems)

    **Perfect for:** Advanced prep, FAANG interviews, senior roles

    **Time per problem:** 60-75 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | **Twitter Feed** | Social | ⭐⭐⭐⭐⭐ | Fan-out, timeline generation |
    | **Instagram** | Social | ⭐⭐⭐⭐⭐ | Photo storage, social graph |
    | **WhatsApp/Chat** | Social | ⭐⭐⭐⭐⭐ | Real-time messaging, presence |
    | **Video Streaming** | Media | ⭐⭐⭐⭐⭐ | CDN, encoding, adaptive bitrate |
    | **Search Engine** | Search | ⭐⭐⭐⭐⭐ | Crawling, indexing, ranking |
    | **E-Commerce Platform** | E-Commerce | ⭐⭐⭐⭐⭐ | Inventory, transactions, catalog |
    | **Ride Sharing** | Location | ⭐⭐⭐⭐⭐ | Geohashing, matching, routing |
    | **News Feed** | Social | ⭐⭐⭐⭐ | Ranking, personalization |
    | **Music Streaming** | Media | ⭐⭐⭐⭐ | Audio delivery, recommendations |
    | **Live Streaming** | Media | ⭐⭐⭐⭐ | Low latency, transcoding |
    | **Video Conferencing** | Media | ⭐⭐⭐⭐ | WebRTC, signaling |
    | **Recommendation System** | Search | ⭐⭐⭐⭐ | Collaborative filtering, ML |
    | **Cloud Storage** | Storage | ⭐⭐⭐⭐ | Sync, conflict resolution |
    | **Payment System** | E-Commerce | ⭐⭐⭐⭐ | Transactions, idempotency |
    | **Ticket Booking** | E-Commerce | ⭐⭐⭐⭐ | Concurrency, locking |
    | **Food Delivery** | E-Commerce | ⭐⭐⭐⭐ | Matching, real-time tracking |
    | **Google Maps** | Location | ⭐⭐⭐⭐ | Routing, traffic algorithms |
    | **Google Docs** | Collaboration | ⭐⭐⭐⭐ | CRDT, real-time collaboration |
    | **Distributed Cache** | Infrastructure | ⭐⭐⭐⭐ | Consistent hashing, replication |
    | **Message Queue** | Infrastructure | ⭐⭐⭐⭐ | Partitioning, ordering |
    | **Code Repository** | Collaboration | ⭐⭐⭐ | Version control, merge |
    | **Analytics Platform** | Monitoring | ⭐⭐⭐⭐ | Event tracking, aggregation |
    | **Distributed Tracing** | Monitoring | ⭐⭐⭐ | Trace IDs, correlation |

=== "🏢 By Company"

    ## FAANG Companies

    ### Meta (Facebook)

    **Focus:** Social graphs, real-time systems, massive scale

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | **Instagram** | 🔴 Hard | Core product, photo storage, feeds |
    | **WhatsApp** | 🔴 Hard | Messaging at scale, real-time |
    | **News Feed** | 🔴 Hard | Timeline generation, ranking |
    | **Twitter Feed** | 🔴 Hard | Fan-out, social graph |
    | **Notification System** | 🟡 Medium | Cross-platform notifications |
    | **Live Streaming** | 🔴 Hard | Facebook Live, Instagram Live |
    | **Chat System** | 🔴 Hard | Messenger architecture |

    ### Amazon

    **Focus:** E-commerce, consistency, API design, transactions

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | **E-Commerce Platform** | 🔴 Hard | Core business, inventory management |
    | **Payment System** | 🔴 Hard | Transactions, consistency |
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | API design fundamentals |
    | **Rate Limiter** | 🟡 Medium | API protection, throttling |
    | **Recommendation System** | 🔴 Hard | Product recommendations |
    | **Distributed Cache** | 🔴 Hard | Performance optimization |
    | **Search Engine** | 🔴 Hard | Product search |

    ### Apple

    **Focus:** Mobile-first, sync, privacy, user experience

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | **iMessage** | 🔴 Hard | Real-time messaging, encryption |
    | **iCloud Storage** | 🔴 Hard | Sync across devices |
    | **Calendar System** | 🟡 Medium | Sync, conflict resolution |
    | **Music Streaming** | 🔴 Hard | Apple Music architecture |
    | **Location Tracking** | 🟡 Medium | Find My, privacy |
    | **Notification System** | 🟡 Medium | APNs architecture |

    ### Netflix

    **Focus:** Video streaming, CDN, recommendations, global scale

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | **Video Streaming** | 🔴 Hard | Core product, CDN strategy |
    | **Recommendation System** | 🔴 Hard | Content personalization |
    | **Analytics Platform** | 🔴 Hard | User behavior tracking |
    | **Distributed Cache** | 🔴 Hard | Content caching |
    | **API Gateway** | 🟡 Medium | Microservices gateway |
    | **Rate Limiter** | 🟡 Medium | API protection |

    ### Google

    **Focus:** Search, scale, distributed systems, ML

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | **Search Engine** | 🔴 Hard | Core product, indexing, ranking |
    | **Google Maps** | 🔴 Hard | Routing, traffic, geospatial |
    | **YouTube** | 🔴 Hard | Video streaming, recommendations |
    | **Google Docs** | 🔴 Hard | Real-time collaboration |
    | **Google Drive** | 🔴 Hard | Cloud storage, sync |
    | **Autocomplete** | 🟡 Medium | Search suggestions |
    | **Web Crawler** | 🟡 Medium | Indexing the web |
    | **Distributed Cache** | 🔴 Hard | Memcached, performance |

    ---

    ## Other Tech Giants

    ### Microsoft

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | **Teams** | 🔴 Hard | Chat, video conferencing |
    | **OneDrive** | 🔴 Hard | Cloud storage, sync |
    | **Calendar System** | 🟡 Medium | Outlook calendar |
    | **Code Repository** | 🔴 Hard | GitHub, Azure DevOps |
    | **Video Conferencing** | 🔴 Hard | Teams meetings |

    ### Uber

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | **Ride Sharing** | 🔴 Hard | Core product, matching |
    | **Food Delivery** | 🔴 Hard | UberEats, routing |
    | **Google Maps** | 🔴 Hard | Navigation, ETA |
    | **Location Tracking** | 🟡 Medium | Real-time tracking |
    | **Notification System** | 🟡 Medium | Driver/rider notifications |
    | **Payment System** | 🔴 Hard | Payment processing |

    ### Airbnb

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | **Ticket Booking** | 🔴 Hard | Reservation system |
    | **Search Engine** | 🔴 Hard | Property search |
    | **Payment System** | 🔴 Hard | Booking payments |
    | **Calendar System** | 🟡 Medium | Availability calendar |
    | **Recommendation System** | 🔴 Hard | Property recommendations |

    ### LinkedIn

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | **News Feed** | 🔴 Hard | Professional feed |
    | **Twitter Feed** | 🔴 Hard | Timeline generation |
    | **Notification System** | 🟡 Medium | Job alerts, messages |
    | **Search Engine** | 🔴 Hard | Job/people search |
    | **Recommendation System** | 🔴 Hard | Job/connection recommendations |

    ### Twitter

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | **Twitter Feed** | 🔴 Hard | Core product, timeline |
    | **Notification System** | 🟡 Medium | Real-time notifications |
    | **Search Engine** | 🔴 Hard | Tweet search |
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | t.co shortener |
    | **Live Streaming** | 🔴 Hard | Twitter Spaces |

=== "🧩 By Concept"

    **Learn specific system design concepts through relevant problems**

    ## Caching (Most Important!)

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**URL Shortener**](url-shortener.md) | Multi-layer caching, cache invalidation | 🟡 Medium |
    | **Rate Limiter** | Distributed cache, sliding window | 🟡 Medium |
    | **Autocomplete** | Cache warming, prefix caching | 🟡 Medium |
    | **News Feed** | Cache strategy for timelines | 🔴 Hard |
    | **Distributed Cache** | Consistent hashing, replication | 🔴 Hard |
    | **Video Streaming** | CDN caching, edge caching | 🔴 Hard |

    ## Database Sharding & Partitioning

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Twitter Feed** | Shard by user_id, fan-out strategy | 🔴 Hard |
    | **Instagram** | Photo metadata sharding | 🔴 Hard |
    | **E-Commerce** | Product catalog sharding | 🔴 Hard |
    | [**URL Shortener**](url-shortener.md) | Shard by short_code prefix | 🟡 Medium |
    | **WhatsApp** | Message sharding, chat_id routing | 🔴 Hard |

    ## Consistent Hashing

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Distributed Cache** | Hash ring, virtual nodes | 🔴 Hard |
    | **Load Balancer** | Server selection, rebalancing | 🟡 Medium |
    | [**URL Shortener**](url-shortener.md) | Shard routing | 🟡 Medium |
    | **Message Queue** | Partition assignment | 🔴 Hard |

    ## Fan-out Pattern

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Twitter Feed** | Push vs pull, hybrid fan-out | 🔴 Hard |
    | **Instagram** | Photo upload fan-out | 🔴 Hard |
    | **Notification System** | Multi-channel fan-out | 🟡 Medium |
    | **News Feed** | Timeline generation | 🔴 Hard |

    ## Real-time Systems

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **WhatsApp** | WebSocket, message delivery | 🔴 Hard |
    | **Live Streaming** | Low latency, buffering | 🔴 Hard |
    | **Video Conferencing** | WebRTC, peer connections | 🔴 Hard |
    | **Google Docs** | OT/CRDT, conflict resolution | 🔴 Hard |
    | **Location Tracking** | GPS updates, real-time | 🟡 Medium |

    ## Geospatial Systems

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Ride Sharing** | Geohashing, driver matching | 🔴 Hard |
    | **Yelp/Nearby** | Quadtree, geospatial queries | 🟡 Medium |
    | **Google Maps** | Graph algorithms, routing | 🔴 Hard |
    | **Food Delivery** | Route optimization | 🔴 Hard |

    ## CDN & Content Delivery

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Video Streaming** | CDN strategy, edge servers | 🔴 Hard |
    | **Image Hosting** | Image CDN, transformations | 🟡 Medium |
    | **Music Streaming** | Audio delivery | 🔴 Hard |
    | **Cloud Storage** | File distribution | 🔴 Hard |

    ## Message Queues & Async Processing

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Message Queue** | Kafka, partitioning, ordering | 🔴 Hard |
    | **Notification System** | Async delivery, retry logic | 🟡 Medium |
    | **Analytics Platform** | Event streaming, processing | 🔴 Hard |
    | **Web Crawler** | Queue management, priority | 🟡 Medium |

    ## Consistency & Transactions

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Payment System** | ACID, idempotency, ledger | 🔴 Hard |
    | **Ticket Booking** | Optimistic/pessimistic locking | 🔴 Hard |
    | **E-Commerce** | Inventory consistency | 🔴 Hard |
    | **Cloud Storage** | Sync, conflict resolution | 🔴 Hard |

    ## Search & Ranking

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | **Search Engine** | Inverted index, TF-IDF, PageRank | 🔴 Hard |
    | **Autocomplete** | Trie, prefix search | 🟡 Medium |
    | **Recommendation System** | Collaborative filtering, ML | 🔴 Hard |
    | **E-Commerce** | Product ranking | 🔴 Hard |

=== "📅 Learning Path"

    **8-week structured program from beginner to advanced**

    ## Week 1-2: Foundation

    **Goal:** Build fundamentals with easy/medium problems

    **Time commitment:** 2-3 problems per week, 2-3 hours per problem

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | [**URL Shortener**](url-shortener.md) | 1-2 | Code generation, caching basics, capacity estimation | 3h |
    | **Rate Limiter** | 3-4 | Token bucket, sliding window, distributed systems | 2h |
    | **Pastebin** | 5-6 | Text storage, expiration, similar to URL shortener | 2h |
    | **Key-Value Store** | 7 | Basic CRUD, simple storage | 1h |

    **✅ Checkpoint:** Can you explain caching strategies and do capacity estimation?

    ---

    ## Week 3-4: Scale & Distribution

    **Goal:** Learn distribution, partitioning, and scaling patterns

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | **Autocomplete** | 8-9 | Trie data structure, prefix caching | 2h |
    | **Notification System** | 10-11 | Fan-out pattern, multi-channel delivery | 3h |
    | **Web Crawler** | 12-13 | Queue management, distributed coordination | 3h |
    | **Yelp/Nearby** | 14 | Geospatial indexing, quadtree | 2h |

    **✅ Checkpoint:** Can you explain fan-out patterns and sharding strategies?

    ---

    ## Week 5-6: Complex Systems (Hard Problems)

    **Goal:** Tackle FAANG-level problems with multiple components

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | **Twitter Feed** | 15-17 | Timeline generation, hybrid fan-out, massive scale | 4h |
    | **Video Streaming** | 18-20 | CDN, encoding, adaptive bitrate, global scale | 4h |
    | **Ride Sharing** | 21-23 | Geohashing, matching algorithms, real-time | 4h |
    | **Metrics Monitoring** | 24-25 | Time-series DB, aggregation | 2h |

    **✅ Checkpoint:** Can you design systems with 100M+ users?

    ---

    ## Week 7-8: Specialization & Practice

    **Goal:** Deep dive into your target company's domain

    ### Choose Your Track:

    === "Social Media Track"

        | Problem | Focus |
        |---------|-------|
        | **Instagram** | Photo storage, social graph |
        | **WhatsApp** | Real-time messaging |
        | **News Feed** | Ranking, personalization |
        | **Live Streaming** | Low latency video |

    === "E-Commerce Track"

        | Problem | Focus |
        |---------|-------|
        | **E-Commerce Platform** | Inventory, transactions |
        | **Payment System** | Consistency, ledger |
        | **Ticket Booking** | Concurrency, locking |
        | **Search Engine** | Product search |

    === "Media Track"

        | Problem | Focus |
        |---------|-------|
        | **Video Streaming** | CDN, encoding |
        | **Music Streaming** | Recommendations |
        | **Video Conferencing** | WebRTC |
        | **Analytics Platform** | Event tracking |

    === "Infrastructure Track"

        | Problem | Focus |
        |---------|-------|
        | **Distributed Cache** | Consistent hashing |
        | **Message Queue** | Partitioning, ordering |
        | **API Gateway** | Routing, auth |
        | **Load Balancer** | Algorithms |

    **✅ Final Checkpoint:** Ready for interviews! Practice explaining designs out loud.

    ---

    ## Interview Week Prep

    **Goal:** Quick review and polish

    | Day | Activity | Time |
    |-----|----------|------|
    | **-7 days** | Review top 10 most frequent problems | 2h |
    | **-6 days** | Mock interview #1 (with friend/peer) | 1h |
    | **-5 days** | Practice calculations and estimations | 1h |
    | **-4 days** | Review trade-offs and bottlenecks | 2h |
    | **-3 days** | Mock interview #2 | 1h |
    | **-2 days** | Review company-specific problems | 2h |
    | **-1 day** | Light review, rest well | 1h |

---

## How to Practice Each Problem

**Follow this systematic approach:**

### 1. First Attempt (30-45 min)
- Read problem statement only
- Try to design it yourself
- Don't look at the solution
- Draw diagrams on paper/whiteboard
- Calculate capacity estimates

### 2. Study Solution (30-45 min)
- Read through complete solution
- Understand each component's purpose
- Note what you missed
- Study the diagrams

### 3. Identify Gaps (15 min)
- What did you miss?
- Which concepts were new?
- What would you do differently?

### 4. Explain Out Loud (20 min)
- Pretend you're in an interview
- Explain the design from scratch
- Focus on trade-offs

### 5. Review (3 days later)
- Can you still explain it?
- Revisit weak areas
- Practice with variations

---

## Interview Preparation Checklist

### Before Your Interview

**Technical Readiness:**
- [ ] Completed 10+ problems (5 easy/medium, 5 hard)
- [ ] Can do capacity calculations in < 5 minutes
- [ ] Understand caching strategies deeply
- [ ] Know database sharding approaches
- [ ] Can explain CAP theorem with examples
- [ ] Comfortable with trade-off discussions

**Communication Readiness:**
- [ ] Practiced 2+ mock interviews
- [ ] Can think out loud naturally
- [ ] Ask clarifying questions first
- [ ] Draw clear diagrams quickly
- [ ] Handle "what if" questions confidently

**Company-Specific:**
- [ ] Completed 5+ problems from target company list
- [ ] Understand their tech stack
- [ ] Know their scale (users, requests, data)
- [ ] Researched their engineering blog

---

## Study Resources

| Resource | Use For |
|----------|---------|
| [Interview Framework](../interviews/framework.md) | 4-step process, structure |
| [Calculations Guide](../interviews/calculations.md) | Quick capacity estimation |
| [Communication Tips](../interviews/communication.md) | Interview techniques |
| [Common Mistakes](../interviews/common-mistakes.md) | What to avoid |

---

## Quick Reference

### Each Problem Contains:

| Section | Time to Spend | What's Included |
|---------|---------------|-----------------|
| **Step 1: Requirements** | 10-15 min | Functional/non-functional requirements, capacity estimation with calculations |
| **Step 2: High-Level Design** | 15-20 min | Architecture diagrams, API design, database schema, data flow |
| **Step 3: Deep Dive** | 15-20 min | Algorithms with code, caching strategies, optimization techniques |
| **Step 4: Scale & Optimize** | 5-10 min | Bottlenecks, trade-offs, monitoring, reliability patterns |

### Difficulty Legend:
- 🟢 **Easy:** Single-server, simple CRUD, basic caching (5 problems)
- 🟡 **Medium:** Distribution, sharding, moderate scale (25 problems)
- 🔴 **Hard:** Massive scale, complex trade-offs, real-time (20 problems)

### Frequency Legend:
- ⭐⭐⭐⭐⭐ Very High (asked by 5+ companies regularly)
- ⭐⭐⭐⭐ High (asked by 3-4 companies)
- ⭐⭐⭐ Medium (asked by 2 companies)

---

## Getting Started

**Choose your starting point:**

- 🎯 **Interview this week** → "By Company" tab → Pick your company's top 10
- 📚 **Learning from scratch** → "Learning Path" tab → Start Week 1
- 🔍 **Master specific concept** → "By Concept" tab → Pick your topic
- ⚡ **Quick review** → "By Difficulty" tab → Practice easy → medium → hard

**Most popular starting problem:** [URL Shortener](url-shortener.md) ⭐⭐⭐⭐⭐

---

**Ready to begin? Pick your approach above and start practicing!** 🚀
