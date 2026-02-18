# Data Modeling and Schema Design

Your database schema is the contract between your application and its data. Get it right, and queries are fast, code is clean, and the system scales naturally. Get it wrong, and you spend months fighting data inconsistencies, writing convoluted joins, or running expensive migrations on tables with billions of rows.

Data modeling is the discipline of deciding **how to structure your data** — what tables or collections to create, how they relate to each other, and how to organize fields within them. It's not about picking a database engine (that's a separate decision). It's about designing the shape of your data so it serves your application's needs efficiently.

```
The Data Modeling Spectrum:

  Highly Normalized                              Highly Denormalized
  (no duplication)                               (optimized for reads)
       │                                                │
       ▼                                                ▼
  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 3NF     │    │ Balanced │    │ Read-    │    │ Fully    │
  │ Tables  │    │ Design   │    │ Optimized│    │ Flattened│
  │         │    │          │    │          │    │          │
  │ Many    │    │ Strategic│    │ Computed │    │ Single   │
  │ joins   │    │ joins +  │    │ views +  │    │ table /  │
  │ needed  │    │ some     │    │ caches   │    │ document │
  │         │    │ copies   │    │          │    │          │
  └─────────┘    └──────────┘    └──────────┘    └──────────┘
       │              │                │               │
   Banking        E-commerce       Social feed     Analytics
   ERP systems    SaaS apps        News feed       Dashboards
```

Most real-world systems land somewhere in the middle, combining normalized core data with denormalized read paths for performance.

---

## Normalization: Eliminating Redundancy

Normalization is a set of rules for organizing data to **minimize duplication**. Each "normal form" builds on the previous one, progressively eliminating different types of redundancy.

The core insight is simple: every fact should be stored in exactly one place. If a customer's address appears in 50 order rows and they move, you'd need to update all 50 rows. Miss one, and your data is inconsistent.

### First Normal Form (1NF): Atomic Values

Every column holds a single, indivisible value. No lists, no comma-separated strings, no nested structures.

```
BEFORE (violates 1NF):
┌────────┬──────────────────────────┐
│ user   │ phone_numbers            │
├────────┼──────────────────────────┤
│ Alice  │ 555-0101, 555-0102       │  ← multiple values in one cell
│ Bob    │ 555-0201                 │
└────────┴──────────────────────────┘

AFTER (1NF):
┌────────┬──────────────┐
│ user   │ phone_number │
├────────┼──────────────┤
│ Alice  │ 555-0101     │  ← one value per cell
│ Alice  │ 555-0102     │
│ Bob    │ 555-0201     │
└────────┴──────────────┘
```

**Why it matters:** You can't efficiently query "find everyone with phone number 555-0102" when phone numbers are buried inside comma-separated strings. You'd need `LIKE '%555-0102%'` which can't use indexes and might match `1555-0102`.

### Second Normal Form (2NF): No Partial Dependencies

Every non-key column depends on the **entire** primary key, not just part of it. This only matters for tables with composite primary keys.

```
BEFORE (violates 2NF):
┌──────────┬────────────┬────────────┬───────────────┐
│ order_id │ product_id │ quantity   │ product_name  │
├──────────┼────────────┼────────────┼───────────────┤
│ 1001     │ P1         │ 3          │ Widget        │  ← product_name depends
│ 1001     │ P2         │ 1          │ Gadget        │    only on product_id,
│ 1002     │ P1         │ 5          │ Widget        │    not on the full key
└──────────┴────────────┴────────────┴───────────────┘
  PK = (order_id, product_id)
  product_name depends only on product_id → partial dependency

AFTER (2NF):
Order_Items:                    Products:
┌──────────┬────────────┬─────┐ ┌────────────┬──────────────┐
│ order_id │ product_id │ qty │ │ product_id │ product_name │
├──────────┼────────────┼─────┤ ├────────────┼──────────────┤
│ 1001     │ P1         │ 3   │ │ P1         │ Widget       │
│ 1001     │ P2         │ 1   │ │ P2         │ Gadget       │
│ 1002     │ P1         │ 5   │ └────────────┴──────────────┘
└──────────┴────────────┴─────┘
```

**Why it matters:** Without 2NF, if someone renames "Widget" to "Super Widget," you'd need to update every row that contains product P1. Miss one row, and the same product has two different names.

### Third Normal Form (3NF): No Transitive Dependencies

Every non-key column depends directly on the primary key, not indirectly through another non-key column.

```
BEFORE (violates 3NF):
┌──────────┬──────────┬───────────┬──────────────┐
│ order_id │ customer │ zip_code  │ city         │
├──────────┼──────────┼───────────┼──────────────┤
│ 1001     │ Alice    │ 94105     │ San Francisco│  ← city depends on
│ 1002     │ Bob      │ 10001     │ New York     │    zip_code, not on
│ 1003     │ Alice    │ 94105     │ San Francisco│    order_id directly
└──────────┴──────────┴───────────┴──────────────┘
  order_id → zip_code → city (transitive dependency)

AFTER (3NF):
Orders:                         Zip_Codes:
┌──────────┬──────────┬────────┐ ┌───────────┬──────────────┐
│ order_id │ customer │ zip    │ │ zip_code  │ city         │
├──────────┼──────────┼────────┤ ├───────────┼──────────────┤
│ 1001     │ Alice    │ 94105  │ │ 94105     │ San Francisco│
│ 1002     │ Bob      │ 10001  │ │ 10001     │ New York     │
│ 1003     │ Alice    │ 94105  │ └───────────┴──────────────┘
└──────────┴──────────┴────────┘
```

### When to Normalize

!!! tip "Rule of Thumb"
    Normalize your **source of truth** (transactional data), then denormalize your **read paths** (caches, search indexes, materialized views). This gives you correctness where it matters and performance where users feel it.

**Normalize when:**

- Data integrity is critical (financial systems, healthcare, inventory)
- Write-heavy workloads where updates to shared data are frequent
- Storage is a concern and you want to avoid duplication
- You need flexible querying — normalized schemas support ad-hoc queries well

**Skip full normalization when:**

- Read performance is the primary concern (analytics dashboards)
- Data is naturally hierarchical (a blog post with its comments)
- The data is immutable (event logs, audit trails — no update anomalies possible)

---

## Denormalization: Trading Writes for Reads

Denormalization is the deliberate introduction of redundancy to improve read performance. Instead of joining three tables at query time, you store the pre-joined result directly.

```
Normalized (3 table join at read time):

  SELECT o.id, c.name, p.title
  FROM orders o
  JOIN customers c ON o.customer_id = c.id
  JOIN products p ON o.product_id = p.id
  WHERE o.id = 1001;

  3 tables touched · index lookups on each · ~2ms

Denormalized (single read):

  SELECT id, customer_name, product_title
  FROM order_details
  WHERE id = 1001;

  1 table · 1 index lookup · ~0.3ms
```

The read is 5-7x faster, but every time a customer changes their name or a product is renamed, you must update every row that contains that data. This is the fundamental trade-off.

### The Read/Write Trade-off

| Aspect | Normalized | Denormalized |
|--------|-----------|--------------|
| **Read speed** | Slower (joins required) | Faster (pre-joined) |
| **Write speed** | Faster (update one place) | Slower (update many places) |
| **Storage** | Minimal (no duplication) | More (data copied) |
| **Consistency** | Guaranteed by schema | Application must maintain |
| **Query flexibility** | High (any join possible) | Low (optimized for specific queries) |
| **Complexity** | In queries (join logic) | In writes (sync logic) |

### Common Denormalization Strategies

**Duplicating columns.** Copy `customer_name` into the orders table. Saves a join on every order lookup. Accept that name changes require updating all their orders.

**Pre-computed aggregates.** Store `total_orders` and `total_spent` on the customer record instead of computing `COUNT(*)` and `SUM(amount)` on every request. Update these counters on each new order.

**Materialized views.** The database automatically maintains a denormalized "table" from a query definition. PostgreSQL supports this natively — you define the query once, and the database rebuilds the view on a schedule or on-demand.

**Caching layers.** Store denormalized data in Redis or Memcached rather than in the database itself. This keeps the source of truth normalized while serving reads from a fast cache.

!!! warning "Denormalization Pitfall"
    The biggest risk is **inconsistency**. When the same fact exists in multiple places, updates must touch all of them. If your application writes to the orders table but forgets to update the customer's `total_spent` counter, the data drifts silently. Always audit your write paths when denormalizing.

---

## Schema Design Patterns by Database Type

### Relational: Entity-Relationship Modeling

Relational design starts by identifying **entities** (things with identity) and **relationships** (how entities connect). This maps directly to tables with foreign keys.

```
Entity-Relationship Diagram — E-Commerce:

  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
  │  Customers   │       │   Orders     │       │  Products    │
  ├──────────────┤       ├──────────────┤       ├──────────────┤
  │ id       PK  │──┐    │ id       PK  │    ┌──│ id       PK  │
  │ name         │  │    │ customer_id FK│◄───┘  │ name         │
  │ email        │  └───►│ total        │       │ price        │
  │ created_at   │       │ status       │       │ category_id FK│──┐
  └──────────────┘       │ created_at   │       └──────────────┘  │
                         └──────┬───────┘                         │
                                │                           ┌─────┴────────┐
                         ┌──────┴───────┐                   │  Categories  │
                         │ Order_Items  │                   ├──────────────┤
                         ├──────────────┤                   │ id       PK  │
                         │ order_id  FK │                   │ name         │
                         │ product_id FK│                   └──────────────┘
                         │ quantity     │
                         │ unit_price   │
                         └──────────────┘
```

**Key concepts:**

- **Primary key (PK):** Uniquely identifies each row. Usually a sequential integer or UUID.
- **Foreign key (FK):** A column that references another table's primary key. Enforces referential integrity — you can't create an order for a customer that doesn't exist.
- **Junction table:** Resolves many-to-many relationships. An order has many products; a product appears in many orders. The `order_items` table sits between them, holding one row per order-product pair.

**One-to-many:** A customer has many orders. The "many" side (orders) holds the foreign key pointing to the "one" side (customers).

**Many-to-many:** Products and orders connect through a junction table (order_items). Each row in the junction table represents one link.

**One-to-one:** Rare, but used for table splitting — putting rarely-accessed columns (like full bio text) in a separate table to keep the main table's rows small and cache-friendly.

### Document: Embedding vs. Referencing

In document databases like MongoDB, you have a fundamental choice for related data: **embed it inside the parent document** or **store it separately and reference it** by ID.

```
EMBEDDED (denormalized):
{
  "order_id": 1001,
  "customer": {
    "name": "Alice",              ← customer data copied into order
    "email": "alice@example.com"
  },
  "items": [
    { "product": "Widget", "qty": 3, "price": 9.99 },
    { "product": "Gadget", "qty": 1, "price": 24.99 }
  ]
}

REFERENCED (normalized):
{
  "order_id": 1001,
  "customer_id": "cust_42",      ← just the ID, fetch separately
  "item_ids": ["item_1001_1", "item_1001_2"]
}
```

Embedding gives you the complete order in a single read — no joins, no additional queries. But if Alice changes her email, you must update it in every order document that embedded her data. Referencing keeps a single source of truth for Alice's data, but reading an order now requires multiple queries.

```
Decision Tree — Embed vs. Reference:

  Is the related data accessed together
  with the parent >80% of the time?
         │
    ┌────┴────┐
    Yes       No
    │         │
    │    Does the related data
    │    change frequently?
    │         │
    │    ┌────┴────┐
    │    Yes       No
    │    │         │
    │    │    Is the related data
    │    │    small and bounded?
    │    │         │
    │    │    ┌────┴────┐
    │    │    Yes       No
    │    │    │         │
    ▼    ▼    ▼         ▼
  EMBED  REF  EMBED    REF

  Examples:
  EMBED: Order line items, blog post comments (bounded)
  REF:   User profiles (change often), product catalog (shared)
```

!!! info "The 16MB Document Limit"
    MongoDB documents have a 16MB size limit. If embedded arrays can grow unboundedly (like comments on a viral post), you'll hit this ceiling. Unbounded arrays should almost always be referenced, not embedded.

### Wide-Column: Row Key and Column Family Design

Wide-column stores like Cassandra and HBase organize data by **row key** and **column families**. The row key determines which node stores the data (similar to a shard key), and column families group related columns together.

```
Wide-Column Layout — Sensor Data:

Row Key: device_id#date
Column Family: "readings"

┌─────────────────────┬──────────────────────────────────────────────┐
│ Row Key             │ readings                                     │
│                     │ temp:08:00  temp:08:05  temp:08:10  hum:08:00│
├─────────────────────┼──────────────────────────────────────────────┤
│ sensor_42#2026-02-17│ 22.5        22.7        22.6        45.2    │
│ sensor_42#2026-02-16│ 21.3        21.5        21.8        44.1    │
│ sensor_99#2026-02-17│ 18.2        18.4        18.3        62.0    │
└─────────────────────┴──────────────────────────────────────────────┘

  ↑ Row key = partition key · Determines which node stores this row
  ↑ Columns are dynamic — each row can have different columns
  ↑ Columns within a row are sorted — efficient range scans
```

**Row key design principles:**

- The row key is your **primary access pattern**. In Cassandra, you can only efficiently query by row key (partition key). Everything else requires a full cluster scan.
- Compound keys like `device_id#date` allow querying all readings for a device on a given day with a single partition read.
- Avoid "hot" row keys. If you use just `date` as the row key, all writes for today hit the same partition. Prepending `device_id` spreads writes across the cluster.

**Column family design:** Group columns that are queried together. If your dashboard reads temperature and humidity together but processes raw voltage data separately, put them in different column families. Each column family is stored in its own set of files on disk, so reading one family doesn't require loading the other.

---

## Access Pattern-Driven Design

The most common mistake in data modeling is designing for the data's structure rather than for how it will be queried. Relational databases let you get away with this — you can join anything to anything. NoSQL databases do not — if you didn't design for a query, that query may be impossible or prohibitively expensive.

```
Traditional approach (model the domain, query later):

  "We have users, products, and orders.
   Let's create tables for each and figure out queries later."

Access-pattern approach (start from queries):

  "Our app needs to:
   1. Show a user's recent orders          → query by user_id, sort by date
   2. Show all items in an order           → query by order_id
   3. Show trending products this week     → query by category + date range
   4. Show a seller's revenue dashboard    → aggregate by seller_id + month

   Now design tables/collections that serve these queries efficiently."
```

!!! tip "NoSQL Golden Rule"
    In NoSQL databases, you should have a table (or collection) design for each major access pattern. If you need data organized two different ways, you may need two tables with the same data in different shapes. This feels wasteful coming from SQL, but it's the intended design pattern.

**Step-by-step process:**

1. **List all queries** your application will make, ranked by frequency
2. **Identify the primary key** for each query — what field does it filter on?
3. **Determine sort order** — do results need to be sorted by date, relevance, or something else?
4. **Group queries** that can be served by the same table structure
5. **Design tables/collections** that match these groups
6. **Accept duplication** — if the same data must be queried two different ways, store it twice

**Example — Chat application:**

```
Access patterns:
  1. "Show all messages in channel X"        → partition: channel_id, sort: timestamp
  2. "Show all channels for user Y"          → partition: user_id
  3. "Search messages containing 'deploy'"   → full-text search index
  4. "Show unread count per channel for Y"   → partition: user_id

Table design (Cassandra/DynamoDB style):

  messages_by_channel:   PK = channel_id, SK = timestamp
  channels_by_user:      PK = user_id, SK = last_activity
  unread_counts:         PK = user_id, SK = channel_id
  (Search handled by Elasticsearch, not the primary database)
```

The same message data appears in `messages_by_channel` (for reading conversations) and feeds the search index. This duplication is intentional and expected.

---

## Real-World Examples

### E-Commerce Product Catalog (Relational)

Amazon's product catalog is a textbook case for relational modeling. Products have strict attributes (price, SKU, stock count) that must be consistent across the entire system — you can't show different prices on the product page and the checkout page.

```
Schema — Product Catalog:

  products              product_variants         product_images
  ┌──────────────┐      ┌──────────────────┐     ┌────────────────┐
  │ id        PK │──┐   │ id            PK │     │ id          PK │
  │ name         │  │   │ product_id    FK │◄─┐  │ variant_id  FK │
  │ description  │  └──►│ sku              │  │  │ url            │
  │ category_id  │      │ color            │  │  │ position       │
  │ brand        │      │ size             │  │  └────────────────┘
  │ base_price   │      │ price_override   │  │
  └──────────────┘      │ stock_count      │  │  product_attributes
                        └──────────────────┘  │  ┌────────────────┐
                                              │  │ product_id  FK │
                                              └──│ attr_name      │
                                                 │ attr_value     │
                                                 └────────────────┘

  Key design choices:
  - Variants (size/color combos) are separate rows, not JSON arrays
  - Each variant has its own stock count and optional price override
  - Attributes use EAV (Entity-Attribute-Value) for flexible properties
  - Images link to variants, not just products (different colors = different photos)
```

**Scale:** Amazon manages ~350 million products. The normalized schema ensures that a price change to a product updates exactly one row, and all pages referencing it reflect the change immediately.

### Social Media User Profiles (Document)

User profiles are a natural fit for document databases. Each profile is self-contained — you rarely join a profile with another profile. The schema varies per user (some have work history, some don't), and the data is read far more often than it's written.

```
Document — User Profile (MongoDB):

{
  "_id": "user_8472",
  "username": "alice_dev",
  "display_name": "Alice Chen",
  "bio": "Building things at Scale Co.",
  "avatar_url": "https://cdn.example.com/avatars/8472.jpg",
  "stats": {
    "followers": 12453,
    "following": 342,
    "posts": 1893
  },
  "work": [
    { "company": "Scale Co.", "role": "Staff Engineer", "current": true },
    { "company": "StartupX", "role": "Senior Engineer", "years": "2022-2024" }
  ],
  "settings": {
    "theme": "dark",
    "notifications": { "email": true, "push": false }
  }
}
```

**Why document works here:**

- A profile is fetched as a whole unit — one read returns everything the profile page needs
- Each profile can have different optional fields (work history, education, portfolio links)
- Stats counters (`followers`, `posts`) are denormalized for fast display — updated asynchronously
- Nested objects (`settings`, `work`) map directly to UI components

**Scale:** Instagram stores ~2 billion user profiles. The profile document is read millions of times per second but written only when the user edits their profile (rare). Counters like follower count are updated asynchronously and may be slightly stale — users tolerate "12.4K followers" being a few seconds behind.

### Time-Series Sensor Data (Wide-Column)

IoT sensor data is append-heavy, time-ordered, and queried by device + time range. Wide-column stores like Cassandra excel here because the row key naturally partitions data by device, and columns sort by timestamp.

```
Cassandra Table — Sensor Readings:

  CREATE TABLE sensor_readings (
    device_id   TEXT,
    day         DATE,
    timestamp   TIMESTAMP,
    temperature DOUBLE,
    humidity    DOUBLE,
    PRIMARY KEY ((device_id, day), timestamp)
  ) WITH CLUSTERING ORDER BY (timestamp DESC);

  Partition key: (device_id, day) → one partition per device per day
  Clustering key: timestamp       → readings sorted within partition

  Query patterns served:
  ✅ "All readings for sensor_42 on 2026-02-17" → single partition read
  ✅ "Last 100 readings for sensor_42 today"    → partition + LIMIT
  ✅ "Temperature between 2pm and 3pm"          → range scan within partition
  ❌ "All sensors with temperature > 30"         → full cluster scan (avoid)
```

**Why this design works:** By including `day` in the partition key, each partition contains at most one day of data per sensor. This prevents unbounded partition growth (a sensor reporting every 5 seconds generates ~17,000 readings per day, which is a manageable partition size). It also means old data naturally falls into its own partitions, making time-based data retention straightforward — just delete old partitions.

**Scale:** Industrial IoT deployments commonly handle 100,000+ sensors writing every few seconds. At 100K sensors x 12 readings/minute, that's 1.2 million writes per second — well within Cassandra's capabilities across a modest cluster.

### Instagram: Denormalized Feed Storage

Instagram's home feed is one of the most demanding read paths in any application — hundreds of millions of users opening the app and expecting to see a personalized feed instantly. The underlying data is normalized (users, posts, follows), but the feed itself is denormalized.

```
How Instagram serves the feed:

  Normalized Source of Truth:
  ┌─────────┐    ┌──────────┐    ┌─────────────┐
  │ users   │    │ posts    │    │ follows     │
  │ 2B rows │    │ 100B+    │    │ 500B+ edges │
  └─────────┘    └──────────┘    └─────────────┘

  Query: "Show Alice's feed"
  = SELECT posts FROM users Alice follows, ORDER BY relevance
  = Join follows (500B rows) with posts (100B rows) → impossibly slow

  Solution: Pre-compute and denormalize

  ┌────────────────────────────────────┐
  │ Feed Cache (per user)              │
  │                                    │
  │ user: alice                        │
  │ feed: [post_id_99, post_id_87,     │
  │        post_id_72, post_id_65...]  │
  │                                    │
  │ Each post_id maps to a cached      │
  │ post object with all display data  │
  └────────────────────────────────────┘

  On post creation: Fan-out to followers' feed caches
  On feed open: Read pre-built feed from cache → <50ms
```

**The trade-off:** When a user with 10 million followers posts a photo, that post ID must be written to 10 million feed caches (fan-out on write). This is expensive on writes but makes reads instant. For celebrity accounts, Instagram uses a hybrid approach — fan-out on write for normal users, fan-out on read (compute at request time) for accounts with millions of followers.

### Uber: Polyglot Persistence

Uber demonstrates that large systems rarely use a single database model. Different parts of the system have different data modeling needs, so they use different databases — each chosen and modeled for its specific access pattern.

```
Uber's Polyglot Persistence:

  ┌─────────────────────────────────────────────────────┐
  │                    Uber Platform                     │
  ├──────────────┬──────────────┬────────────┬──────────┤
  │ User/Trip    │ Geospatial   │ Real-time  │ Analytics│
  │ Data         │ Matching     │ Pricing    │          │
  ├──────────────┼──────────────┼────────────┼──────────┤
  │ MySQL        │ Redis +      │ In-memory  │ Hive +   │
  │ (sharded)    │ Custom index │ (Flink)    │ Presto   │
  │              │              │            │          │
  │ Normalized   │ Geo-hashed   │ Streaming  │ Star     │
  │ relational   │ key-value    │ events     │ schema   │
  │ model        │ model        │            │ (OLAP)   │
  │              │              │            │          │
  │ ACID for     │ "Drivers     │ "Surge     │ "Revenue │
  │ payments     │ near (lat,   │ multiplier │ by city  │
  │ and trips    │ lng)"        │ right now" │ by month"│
  └──────────────┴──────────────┴────────────┴──────────┘

  Each service owns its data model independently.
  No single "universal" schema — each is optimized for its workload.
```

**Key insight:** Uber's trip data is relationally modeled in sharded MySQL (ACID transactions for payments). Their driver location data uses geospatial indexes optimized for "find all drivers within 2km" queries. Their analytics uses a denormalized star schema for fast aggregation. Trying to force all of this into one data model would make everything mediocre.

---

## Common Mistakes in Data Modeling

### 1. Premature Denormalization

Denormalizing before you have evidence of a performance problem. Start normalized. Profile your queries. Denormalize only the specific paths that are too slow.

**Symptom:** Complex write paths that maintain denormalized copies, but the read performance was never actually a bottleneck.

### 2. Treating NoSQL Like SQL

Designing a document database with dozens of small collections that reference each other by ID — essentially recreating a relational schema without the joins. If your documents are tiny and always need to be fetched together, you've lost the benefit of documents.

```
Anti-pattern — relational thinking in MongoDB:

  orders:      { order_id: 1, customer_id: 42 }
  customers:   { customer_id: 42, name: "Alice" }
  addresses:   { address_id: 7, customer_id: 42, street: "..." }
  order_items: { order_id: 1, product_id: 99, qty: 3 }
  products:    { product_id: 99, name: "Widget" }

  Loading an order page = 5 separate queries
  → You've built a relational database, but worse (no joins)

  Better — embed what's needed:
  {
    order_id: 1,
    customer: { name: "Alice", address: "..." },
    items: [{ name: "Widget", qty: 3, price: 9.99 }]
  }

  Loading an order page = 1 query
```

### 3. Ignoring Access Patterns

Designing a beautiful, normalized schema and then discovering your most common query requires joining 7 tables. The schema is technically correct but practically unusable.

**Fix:** Write your top 10 queries before designing the schema. Then design the schema to serve those queries efficiently.

### 4. Unbounded Growth in Documents

Embedding arrays that grow without limit — comments on a post, events in a log, items in a cart. The document grows over time, eventually hitting size limits or degrading performance as the database rewrites the entire document on each append.

**Fix:** If an array can grow beyond a few hundred items, store it as a separate collection with a reference.

### 5. Using Entity-Attribute-Value (EAV) Everywhere

The EAV pattern (storing arbitrary key-value pairs as rows) is useful for truly dynamic attributes but terrible for data you query frequently. Each "column" becomes a row, turning a simple `WHERE color = 'red'` into a join with filtering.

**Use EAV for:** Product attributes that vary by category (shoes have "sole type," laptops have "screen size").

**Don't use EAV for:** Core fields that every record shares and that you filter on regularly.

### 6. Not Planning for Schema Evolution

Schemas change. Products gain new fields, relationships change shape, new features require new data. If your initial design doesn't account for evolution, you'll face painful migrations.

**Relational:** Use `ALTER TABLE` for additive changes (new nullable columns are cheap). Plan for database migration tooling from day one.

**Document:** Use schema versioning — include a `schema_version` field and handle old and new formats in application code.

---

## Normalized vs. Denormalized: Complete Trade-off Comparison

| Dimension | Normalized | Denormalized |
|-----------|-----------|--------------|
| **Read latency** | Higher (joins at query time) | Lower (pre-joined data) |
| **Write latency** | Lower (update one place) | Higher (update many places) |
| **Storage cost** | Lower (no duplication) | Higher (data copied) |
| **Data consistency** | Strong (single source of truth) | Eventual (must sync copies) |
| **Write complexity** | Simple (single table writes) | Complex (fan-out updates) |
| **Read complexity** | Complex (multi-table joins) | Simple (single read) |
| **Schema flexibility** | High (joins support any query) | Low (optimized for known queries) |
| **Scaling reads** | Add read replicas | Already fast; cache further |
| **Scaling writes** | Straightforward | Complex (more copies = more writes) |
| **Best for** | OLTP, financial, source of truth | OLAP, feeds, dashboards, caches |
| **Example** | Banking ledger | Social media feed |
| **Database fit** | PostgreSQL, MySQL | Redis, Cassandra, DynamoDB |

!!! note "They're Not Mutually Exclusive"
    Most production systems use both. The transactional database stays normalized (the source of truth), while read-optimized views, caches, and search indexes hold denormalized copies. Change Data Capture (CDC) tools like Debezium can automatically propagate changes from the normalized source to denormalized targets.

---

## Decision Framework: How to Choose Your Model

When you're starting a new project or designing a new data domain, work through these questions in order:

```
Step 1: What are your access patterns?
        ┌──────────────────────────────────────┐
        │ List your top 5-10 queries by volume │
        │ What fields do they filter on?       │
        │ What fields do they sort by?         │
        │ How many results do they return?     │
        └──────────────┬───────────────────────┘
                       ▼
Step 2: What are your consistency requirements?
        ┌──────────────────────────────────────┐
        │ Financial / inventory → strong ACID  │
        │ Social / content → eventual is OK    │
        │ Mixed → normalize core, cache rest   │
        └──────────────┬───────────────────────┘
                       ▼
Step 3: What is your read-to-write ratio?
        ┌──────────────────────────────────────┐
        │ Read-heavy (>90% reads)?             │
        │  → Denormalize aggressively          │
        │ Write-heavy (>50% writes)?           │
        │  → Normalize to reduce write fan-out │
        │ Balanced?                            │
        │  → Normalize core, cache reads       │
        └──────────────┬───────────────────────┘
                       ▼
Step 4: What is the shape of your data?
        ┌──────────────────────────────────────┐
        │ Highly relational (many joins)?      │
        │  → Relational database               │
        │ Self-contained entities?             │
        │  → Document database                 │
        │ Time-ordered, append-only?           │
        │  → Wide-column or time-series DB     │
        │ Graph-shaped (relationships are key)?│
        │  → Graph database                    │
        └──────────────┬───────────────────────┘
                       ▼
Step 5: What is your scale?
        ┌──────────────────────────────────────┐
        │ < 10M records, < 1K QPS?             │
        │  → Almost any model works. Choose    │
        │    what your team knows best.        │
        │ 10M-1B records, 1K-100K QPS?         │
        │  → Model choice matters. Optimize    │
        │    for your primary access pattern.  │
        │ > 1B records, > 100K QPS?            │
        │  → You'll likely need polyglot       │
        │    persistence (multiple databases). │
        └──────────────────────────────────────┘
```

**Quick Reference — Model by Use Case:**

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Banking transactions | Normalized relational | ACID, audit trail, regulatory compliance |
| Product catalog | Relational + search index | Structured data + full-text search |
| User profiles | Document | Self-contained, variable schema |
| Social media feed | Denormalized cache | Read-heavy, latency-sensitive |
| Chat messages | Wide-column | Time-ordered, partition by channel |
| IoT sensor data | Wide-column / time-series | Append-heavy, time-range queries |
| Recommendation engine | Graph + document | Relationship traversal + user data |
| Analytics dashboard | Star schema (denormalized) | Aggregation-optimized |
| Shopping cart | Document or key-value | Short-lived, self-contained |
| Audit log | Append-only (immutable) | Write-once, time-ordered scans |

---

## Further Reading

**Related Topics:**

- [Database Types](database-types.md) — When to use each database type
- [Indexing](indexing.md) — Make your queries fast after modeling your data
- [Sharding](sharding.md) — Horizontal partitioning when one machine isn't enough
- [Replication](replication.md) — Read replicas and high availability
- [Caching Strategies](../caching/strategies.md) — Caching as a denormalization layer
- [Data Consistency](../../fundamentals/data-consistency.md) — Consistency models and trade-offs
