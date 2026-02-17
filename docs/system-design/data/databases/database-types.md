# Database Types

When someone asks "which database should I use?", the honest answer is always "it depends on your workload." A database optimized for processing financial transactions is fundamentally different from one designed for full-text search or real-time analytics. Understanding the landscape of database types — what each one is built for, how it stores data, and where it falls short — is essential for making good architectural decisions.

This guide walks through every major database type, explains the thinking behind each one, and ends with a quick-reference table you can use when making decisions.

---

## Relational Databases (SQL)

Relational databases are the foundation of modern data management. They organize data into **tables** with rows and columns, enforce relationships through foreign keys, and guarantee consistency through ACID transactions. You interact with them using SQL — a language that's been the industry standard for 40+ years.

The core idea is simple: define your schema upfront (what columns exist, what types they are, what constraints apply), and the database enforces those rules on every write. If you say `email` must be unique, the database rejects duplicates. If you say `order.customer_id` must reference an existing customer, the database won't let you create orphan orders.

### When Relational Databases Shine

**Complex queries with joins.** "Show me all orders from customers in Texas who spent more than $500 last month, grouped by product category." SQL handles this in a single query. Doing this across denormalized documents or key-value pairs requires significant application logic.

**Transactions that span multiple tables.** Transfer money between accounts? That's a debit from one row and a credit to another — both must succeed or both must fail. ACID transactions guarantee this atomically, even if the server crashes mid-operation.

**Data integrity matters.** Financial systems, healthcare records, inventory management — anywhere incorrect data causes real harm. Foreign keys, constraints, and strong typing catch errors before they're stored.

### Where They Struggle

**Flexible or evolving schemas.** Adding a column to a 500-million-row table can be a multi-hour operation that locks writes. If your data model changes weekly (as in early-stage startups), the rigidity of SQL schemas becomes friction.

**Horizontal scaling for writes.** Relational databases scale reads well (add replicas), but scaling writes requires sharding, which breaks the very features that make SQL powerful — joins, transactions, and foreign keys don't work across shards without significant complexity.

### The Major Players

**PostgreSQL** — the most feature-rich open-source database. Supports JSON, full-text search, geospatial (PostGIS), and custom extensions. The default choice for most new applications.

**MySQL** — dominant in web applications. Powers much of the internet (WordPress, Facebook's early years). Excellent read performance, strong replication ecosystem.

**SQL Server** — Microsoft's enterprise database. Deep integration with .NET and Azure. Common in enterprise and business intelligence.

**Oracle** — the original enterprise database. Massive feature set, extremely expensive. Found in banks, governments, and large enterprises.

---

## Document Databases

Document databases store data as **flexible, JSON-like documents** rather than rigid rows and columns. Each document can have a different structure — one product might have a `color` field while another has a `size_chart` array. There's no need to define a schema upfront or run migrations when your data model changes.

```
Relational (rigid):
┌────┬──────────┬───────┬───────┐
│ id │ name     │ color │ size  │
├────┼──────────┼───────┼───────┤
│ 1  │ T-Shirt  │ blue  │ M     │
│ 2  │ Laptop   │ NULL  │ NULL  │  ← wasted columns
└────┴──────────┴───────┴───────┘

Document (flexible):
{ "id": 1, "name": "T-Shirt", "color": "blue", "size": "M" }
{ "id": 2, "name": "Laptop", "cpu": "M3", "ram": "16GB", "ports": ["USB-C", "HDMI"] }
  ↑ Each document has exactly the fields it needs
```

Documents map naturally to objects in application code — a JSON document returned from MongoDB looks almost identical to the object your JavaScript or Python code works with. No ORM impedance mismatch.

### When Document Databases Shine

**Varying structures within the same collection.** An e-commerce catalog where shoes have `shoe_size` and laptops have `cpu_speed`. In SQL, you'd either have a single table with dozens of nullable columns or a complex EAV (entity-attribute-value) pattern. In a document database, each product simply has the fields it needs.

**Rapid development and evolving schemas.** Early-stage applications where the data model changes frequently. No migrations, no downtime for schema changes.

**Embedded data that's read together.** If you always read a blog post with its comments, storing comments as an embedded array inside the post document means one read instead of a join.

### Where They Struggle

**Complex relationships and joins.** If your queries frequently need data from multiple collections (like "all orders for customers in Texas"), you'll either denormalize (duplicate data) or do multiple queries and join in application code. Neither is as clean as a SQL JOIN.

**Multi-document transactions.** MongoDB added multi-document transactions in v4.0, but they're slower and more limited than SQL transactions. If your workload is transaction-heavy, SQL is still the better fit.

### The Major Players

**MongoDB** — the dominant document database. Rich query language, aggregation pipeline, Atlas managed service. Used by companies from startups to Fortune 500.

**CouchDB** — built for offline-first applications with multi-master replication. Each node can operate independently and sync later.

**Amazon DocumentDB** — MongoDB-compatible managed service on AWS.

**Firestore** — Google's serverless document database, popular for mobile and web apps.

---

## Key-Value Stores

Key-value stores are the simplest database model: a unique key maps to a value. Think of it as a giant hash table. You GET by key, PUT by key, DELETE by key. That's it.

This simplicity is the point — by giving up complex queries, you get **extraordinary speed**. Redis, the most popular key-value store, serves reads in under 1 millisecond because everything lives in memory.

### When Key-Value Stores Shine

**Caching.** The most common use case by far. Put frequently-accessed database results, API responses, or computed values in Redis. The application checks Redis first (sub-millisecond), and only hits the primary database on cache miss.

**Session storage.** User sessions are a perfect fit — keyed by session ID, with automatic expiration (TTL). Redis handles millions of sessions with trivial latency.

**Real-time features.** Leaderboards (Redis sorted sets), rate limiting (atomic counters), pub/sub messaging, and real-time counters.

### Where They Struggle

**Anything beyond key-based access.** "Find all users in Texas" is impossible without scanning every key — there's no query language, no indexes on values. If you need to search or filter, you need a different database.

**Large values or complex data.** While Redis supports lists, sets, and hashes, it's not designed for storing large documents or complex nested structures.

### The Major Players

**Redis** — the gold standard. In-memory with optional persistence. Rich data structures (strings, hashes, lists, sets, sorted sets, streams). Used as cache, message broker, session store, and real-time data store.

**Memcached** — simpler than Redis (strings only), but excellent for pure caching use cases. Multi-threaded, slightly better raw throughput for simple GET/SET.

**DynamoDB** — AWS's managed key-value/document hybrid. Automatically scales, single-digit millisecond latency. Not purely key-value (supports queries on sort keys), but key-based access is its strength.

---

## Column-Family (Wide-Column) Databases

Column-family databases store data in columns rather than rows. This sounds like a minor difference but fundamentally changes performance characteristics. When a relational database reads "average salary of all employees," it loads entire rows (name, address, phone, salary, ...) but only needs the salary column. A column-family database reads only the salary column — far less I/O.

Data is organized into **column families** — groups of columns that are frequently accessed together. Within each column family, data is sorted by a row key, making range queries on that key very efficient.

### When Column-Family Databases Shine

**Write-heavy workloads at massive scale.** Cassandra can handle hundreds of thousands of writes per second with linear scaling — just add nodes. This makes it ideal for event logging, metrics collection, and IoT sensor data.

**Time-series and analytics.** Data naturally organized by time, where you frequently query ranges ("all metrics from the last hour"). The combination of sorted storage and efficient column reads makes time-range queries fast.

**Tunable consistency.** Cassandra lets you choose consistency per query — `ONE` for fast eventual consistency, `QUORUM` for majority agreement, `ALL` for strong consistency. This flexibility lets you optimize per use case within the same database.

### Where They Struggle

**Ad-hoc queries and joins.** Column-family databases require you to model data around your query patterns. If you don't know your access patterns upfront, or if they change frequently, the rigid data modeling becomes painful.

**Small-scale applications.** The operational overhead of running a Cassandra cluster isn't justified for a database with a few million rows. Use PostgreSQL instead.

### The Major Players

**Apache Cassandra** — masterless, linearly scalable, tunable consistency. Powers Apple (400,000+ nodes), Netflix, Discord, and Instagram. The go-to for massive write throughput.

**HBase** — built on Hadoop, provides strong consistency (unlike Cassandra). Used for large-scale analytics in the Hadoop ecosystem.

**Google Bigtable** — Google's managed column-family service. The original inspiration for Cassandra and HBase. Handles petabytes.

**ScyllaDB** — Cassandra-compatible but rewritten in C++ for dramatically better performance per node.

---

## Graph Databases

Most databases model relationships as an afterthought — foreign keys in SQL, embedded references in documents. Graph databases flip this: **relationships are first-class citizens**, stored and queried as efficiently as the data itself.

Data is stored as **nodes** (entities) and **edges** (relationships), each with properties. A social network in a graph database looks exactly like the whiteboard diagram you'd draw: circles for people, lines for friendships, with labels on everything.

```
Graph: Social Network

(Alice)──FRIENDS_WITH──(Bob)
   │                     │
 LIKES                 WORKS_AT
   │                     │
(Post #42)           (Acme Corp)
```

The key advantage: traversing relationships is a **constant-time operation** regardless of dataset size. "Find friends of friends" in SQL requires a self-join that gets exponentially slower as the table grows. In a graph database, it's following pointers — always fast.

### When Graph Databases Shine

**Deep relationship queries.** "Find all friends of friends who also like jazz and live within 50 miles" — queries that traverse multiple hops through relationships. Social networks, recommendation engines, and fraud detection all rely on this.

**Fraud detection.** Identifying suspicious patterns in financial transactions — circular money transfers, shared identifiers across accounts, unusual connection patterns. These are naturally graph problems.

**Knowledge graphs.** Mapping relationships between concepts, entities, and facts. Google's Knowledge Graph, Wikipedia's data model, and enterprise knowledge management systems.

### Where They Struggle

**Simple CRUD with no relationship traversal.** If your queries are "get user by ID" or "list all products," a graph database adds complexity with no benefit.

**High-volume writes.** Graph databases aren't optimized for write throughput. For append-heavy workloads (logs, metrics), use column-family or time-series databases.

### The Major Players

**Neo4j** — the market leader. Cypher query language (intuitive, pattern-matching syntax). Excellent tooling and visualization.

**Amazon Neptune** — AWS managed graph service. Supports both property graph (Gremlin) and RDF (SPARQL) models.

**ArangoDB** — multi-model: graph + document + key-value in one database.

---

## Time-Series Databases

Time-series databases are purpose-built for data that arrives as a stream of timestamped points: server CPU metrics, stock prices, IoT sensor readings, application logs. While you *could* store this in PostgreSQL, a time-series database provides 10-100x better compression and query performance for this specific pattern.

The key optimizations: data is always appended (never randomly updated), compression exploits the fact that consecutive measurements are similar (CPU was 42%, then 43%, then 41%), and built-in functions handle time-based aggregation ("average CPU per 5-minute window").

### When Time-Series Databases Shine

**Infrastructure monitoring.** Every server, container, and service emitting metrics every 10 seconds. Prometheus + Grafana is the industry standard for this.

**IoT and sensor data.** Thousands of devices each sending readings every second. Time-series databases handle millions of inserts per second with automatic compression and retention policies.

**Financial data.** Stock ticks, trading metrics, risk calculations — all timestamped, all queried by time range.

### Where They Struggle

**General-purpose queries.** These databases are optimized for "data between time A and time B." If you need complex joins, full-text search, or relationship traversal, use a different database.

**Frequent updates to existing data.** Time-series data is append-only. Updating historical points is typically slow or unsupported.

### The Major Players

**InfluxDB** — purpose-built time-series database. Excellent performance, built-in retention policies and downsampling.

**Prometheus** — the standard for infrastructure monitoring. Pull-based collection, powerful PromQL query language, deep Kubernetes integration.

**TimescaleDB** — PostgreSQL extension that adds time-series superpowers. Get time-series performance while keeping full SQL compatibility.

**ClickHouse** — column-oriented analytical database, exceptional for time-series analytics at scale. Used by Cloudflare, Uber, and eBay.

---

## Vector Databases

Vector databases are the newest category, driven by the AI/ML boom. They store and search **high-dimensional vectors** — numerical representations of text, images, or audio generated by machine learning models (embeddings).

When you ask ChatGPT a question with RAG (Retrieval-Augmented Generation), your question is converted to a vector, and the vector database finds the most semantically similar documents — not by keyword matching, but by meaning. "How do I fix a flat tire?" would match a document about "changing a punctured wheel" even though they share no keywords.

### When Vector Databases Shine

**Semantic search.** Search by meaning rather than exact keywords. "Articles about climate impact on agriculture" finds relevant results even if those exact words don't appear.

**RAG (Retrieval-Augmented Generation).** Feed LLMs relevant context from your own documents. The vector database retrieves the most relevant chunks, and the LLM generates an answer grounded in your data.

**Recommendation systems.** "Users who liked this also liked..." — represent users and items as vectors, find the nearest neighbors.

**Image/audio similarity.** "Find photos similar to this one" or "find songs that sound like this."

### Where They Struggle

**Traditional queries.** If you need "all users in Texas with orders over $500," use a relational database. Vector databases answer "what's most similar to X," not structured queries.

**Small datasets without ML.** If you have 1,000 documents and keyword search works fine, a vector database is overkill.

### The Major Players

**Pinecone** — fully managed, production-grade. Simple API, excellent scaling.

**Weaviate** — open-source with built-in vectorization (can generate embeddings for you).

**Chroma** — lightweight, developer-friendly. Popular for prototyping and smaller RAG applications.

**pgvector** — PostgreSQL extension. Add vector search to your existing PostgreSQL database without a separate system.

**Milvus** — open-source, designed for billion-scale vector search.

---

## Search Engines

Search engines aren't traditional databases, but they store data and answer queries, so they belong in this guide. They're built around **inverted indexes** — mapping every word to the documents containing it — which makes full-text search orders of magnitude faster than `LIKE '%keyword%'` in SQL.

Beyond basic text search, they provide relevance scoring (BM25), faceted filtering, autocomplete, fuzzy matching, and highlighting. This is why every e-commerce site, documentation portal, and content platform uses a search engine alongside its primary database.

### When Search Engines Shine

**Full-text search with relevance ranking.** "Find the most relevant articles about database sharding" — ranked by how well they match, not just whether they contain the words.

**Log aggregation and analysis.** The ELK stack (Elasticsearch, Logstash, Kibana) is the standard for centralizing and searching application logs.

**Faceted navigation.** "Filter by brand, price range, color, and size" on an e-commerce site — with instant count updates as filters change.

### Where They Struggle

**Primary data storage.** Search engines aren't designed for transactional consistency. Use them as a secondary index alongside a primary database, not as the source of truth.

**Frequent small updates.** Elasticsearch's near-real-time indexing has a ~1 second delay. For data that changes rapidly and must be read immediately, use a database with strong consistency.

### The Major Players

**Elasticsearch** — the dominant search engine. Powers search at Netflix, Uber, GitHub, and Wikipedia. Part of the Elastic Stack (ELK).

**Apache Solr** — built on Lucene (like Elasticsearch). Mature, battle-tested, slightly less developer-friendly.

**Meilisearch / Typesense** — newer, lighter alternatives focused on instant search with typo tolerance. Great for smaller-scale applications.

---

## Choosing the Right Database

Most applications don't use a single database. This pattern — called **polyglot persistence** — uses the best database for each workload:

```
Typical modern application:

PostgreSQL     → primary data (users, orders, products)
Redis          → caching, sessions, rate limiting
Elasticsearch  → search and log aggregation
InfluxDB       → metrics and monitoring
```

Start with PostgreSQL (or MySQL) for your primary data. It handles 90% of use cases well. Add specialized databases only when you have a specific workload that PostgreSQL can't serve efficiently — not because a technology sounds exciting.

```
Decision flow:
    │
    ├─ Structured data, transactions, complex queries
    │   └─ Relational (PostgreSQL, MySQL)
    │
    ├─ Flexible schema, document-oriented, rapid iteration
    │   └─ Document (MongoDB, Firestore)
    │
    ├─ Ultra-fast lookups by key, caching, sessions
    │   └─ Key-Value (Redis, DynamoDB)
    │
    ├─ Massive write throughput, time-series, IoT
    │   └─ Column-Family (Cassandra) or Time-Series (InfluxDB)
    │
    ├─ Deep relationship traversal, fraud detection
    │   └─ Graph (Neo4j, Neptune)
    │
    ├─ Full-text search, log analysis
    │   └─ Search Engine (Elasticsearch)
    │
    └─ AI/ML similarity search, RAG, recommendations
        └─ Vector (Pinecone, pgvector, Weaviate)
```

---

## Quick Reference Table

| Database Type | Data Model | Best For | Not For | Consistency | Scaling | Key Examples |
|---|---|---|---|---|---|---|
| **Relational** | Tables, rows, columns | Transactions, complex queries, data integrity | Rapidly changing schemas, massive write scale | Strong (ACID) | Vertical; horizontal via sharding | PostgreSQL, MySQL, Oracle |
| **Document** | JSON-like documents | Flexible schemas, catalogs, CMS, rapid dev | Heavy joins, complex transactions | Tunable | Horizontal (built-in sharding) | MongoDB, CouchDB, Firestore |
| **Key-Value** | Key → value pairs | Caching, sessions, real-time counters | Complex queries, searching by value | Eventual / Strong | Horizontal | Redis, Memcached, DynamoDB |
| **Column-Family** | Column families, row keys | Write-heavy, time-series, IoT, analytics | Ad-hoc queries, small datasets | Tunable | Horizontal (linear) | Cassandra, HBase, Bigtable |
| **Graph** | Nodes + edges | Relationship traversal, fraud, social, knowledge graphs | Simple CRUD, high write throughput | Strong | Vertical mostly | Neo4j, Neptune, ArangoDB |
| **Time-Series** | Timestamped data points | Monitoring, metrics, IoT sensors, financial ticks | General-purpose queries, frequent updates | Eventual | Horizontal | InfluxDB, Prometheus, TimescaleDB |
| **Vector** | High-dimensional vectors | Semantic search, RAG, recommendations, image similarity | Structured queries, small datasets | Eventual | Horizontal | Pinecone, Weaviate, pgvector |
| **Search Engine** | Inverted index + documents | Full-text search, log analysis, faceted navigation | Primary storage, strong consistency | Near-real-time | Horizontal (sharded) | Elasticsearch, Solr, Meilisearch |

### When to Use What — By Scenario

| Scenario | Primary DB | Supporting DBs | Why |
|---|---|---|---|
| **SaaS Application** | PostgreSQL | Redis (cache), Elasticsearch (search) | Strong consistency for business data, fast cache, good search |
| **E-commerce** | PostgreSQL or MySQL | Redis (sessions/cart), Elasticsearch (product search) | Transactions for orders, flexible search for catalogs |
| **Social Network** | PostgreSQL | Neo4j (graph), Redis (feeds/cache) | Relational for core data, graph for friend-of-friend queries |
| **IoT Platform** | TimescaleDB or Cassandra | Redis (real-time), Grafana+Prometheus (monitoring) | Time-series optimized for sensor data |
| **AI/RAG Application** | PostgreSQL + pgvector | Redis (cache) | Embeddings search alongside relational data |
| **Gaming** | DynamoDB or Redis | PostgreSQL (accounts), InfluxDB (metrics) | Low-latency leaderboards and sessions |
| **Log/Monitoring** | Elasticsearch | Prometheus (metrics), PostgreSQL (config) | Full-text search over billions of log entries |

---

## Key Takeaways

1. **Start with PostgreSQL.** It covers relational data, JSON documents, full-text search, and even vector search (pgvector). Add specialized databases only when you hit a specific limitation.

2. **Choose based on access patterns, not hype.** Every database type exists because it optimizes for a specific workload. Match the database to your actual query patterns.

3. **Polyglot persistence is normal.** Most production systems use 2-4 different databases. PostgreSQL for core data, Redis for caching, Elasticsearch for search, InfluxDB for metrics — each doing what it does best.

4. **The CAP theorem is always in play.** Relational databases prioritize consistency (CP). Cassandra and DynamoDB prioritize availability (AP). Understand what your application can tolerate.

5. **Managed services reduce operational burden.** DynamoDB, Cloud Spanner, Atlas, and ElastiCache handle replication, scaling, and failover. The cost premium is often worth it.

---

## Related Topics

- **[Indexing](indexing.md)** — how databases organize data for fast lookups
- **[Replication](replication.md)** — keeping copies across multiple machines
- **[Sharding](sharding.md)** — splitting data for horizontal scaling
