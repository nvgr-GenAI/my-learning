# Database Indexing

Imagine you have a table with 10 million user records and you need to find the user with email `bob@example.com`. Without any special structure, the database has no choice — it reads row 1, checks, reads row 2, checks, and so on through all 10 million rows. This is called a **full table scan**, and it's exactly as slow as it sounds.

This is the problem that indexing solves.

---

## The Problem: Finding a Needle in a Haystack

When a database stores data, rows are written to disk in the order they arrive — much like tossing papers into a pile. There's no inherent organization. So when you ask "find Bob's record," the database must look at every single row to be sure it hasn't missed anything.

```
Full Table Scan: SELECT * FROM users WHERE email = 'bob@example.com'

Row 1:  alice@example.com   — not a match
Row 2:  charlie@example.com — not a match
Row 3:  dave@example.com    — not a match
...
Row 7,234,891: bob@example.com — FOUND!
...but must continue checking remaining 2,765,109 rows
   (there might be another bob@example.com)

Total: 10,000,000 rows examined · ~2 seconds
```

For a table with 1,000 rows, this is fine. For 10 million rows, it's painfully slow. For 1 billion rows, it's simply not feasible. We need a way to jump directly to the data we want.

---

## What is an Index?

An index is a separate data structure that the database maintains alongside your table. Its job is simple: given a column value, tell the database exactly where to find the matching rows — without scanning the entire table.

**Think of a textbook.** If you want to learn about "B-Trees," you don't read the book cover to cover. You flip to the index at the back, find "B-Trees: page 142," and go directly there. A database index works the same way.

Every index stores two things:

1. **The indexed value** — the column data being indexed (e.g., email addresses)
2. **A pointer** — where to find the actual row on disk (page number + offset)

```
Index on users.email (sorted):
┌──────────────────────┬───────────────┐
│ Email (sorted)       │ Row Location  │
├──────────────────────┼───────────────┤
│ alice@example.com    │ Page 42, Row 3│
│ bob@example.com      │ Page 17, Row 8│
│ charlie@example.com  │ Page 91, Row 1│
│ dave@example.com     │ Page 5, Row 12│
│ ...                  │               │
└──────────────────────┴───────────────┘

Query: WHERE email = 'bob@example.com'
  → Binary search the sorted index → Page 17, Row 8 → fetch the row
```

Because the index is sorted, the database can use binary search instead of scanning everything. For 10 million rows, binary search needs only about 23 comparisons instead of 10 million. That's the difference between 0.1 milliseconds and 2 seconds.

---

## The Fundamental Trade-off

Before going further, you need to understand the one trade-off that governs all indexing decisions:

**Indexes make reads faster and writes slower.**

Every time you INSERT, UPDATE, or DELETE a row, the database must also update every index on that table. A table with 5 indexes means every single write does 6 operations (1 table write + 5 index updates). The more indexes you add, the slower writes become.

|   | Without Index | With Index |
|---|--------------|------------|
| **Reading** | O(n) — scan every row | O(log n) — tree traversal |
| **Writing** | Fast — just append the row | Slower — update table + every index |
| **Storage** | Table data only | Table + index structure (10-30% extra) |

This is why you don't just "index everything." Every index is a conscious trade-off between read speed and write overhead.

---

## How Databases Read Data: Pages and Buffer Pool

Before diving into index types, there's one concept you need to understand — databases don't read individual rows from disk. They read **pages**, which are fixed-size blocks (typically 8 KB in PostgreSQL, 16 KB in MySQL).

```
Data on Disk (organized as pages):
┌──────────┬──────────┬──────────┬──────────┐
│ Page 1   │ Page 2   │ Page 3   │ Page 4   │
│ Rows 1-50│ Rows51-99│Rows 100- │ Rows 151-│
│          │          │   150    │   200    │
└──────────┴──────────┴──────────┴──────────┘

Reading 1 row = reading its entire page from disk.
```

Databases also keep a **buffer pool** — a cache of frequently-accessed pages in RAM. When you access a page, it stays in memory so the next access is nearly instant. This is why index root nodes (accessed on every lookup) are effectively free — they're always in the buffer pool.

This page-based architecture is why some indexes perform dramatically better than others for certain access patterns. Range queries that read consecutive pages (sequential I/O) are 10-100x faster than queries that jump between random pages (random I/O).

---

## Clustered vs Non-Clustered: The Most Important Distinction

Of all the ways to categorize indexes, this is the one that matters most. It determines how your data is **physically stored on disk**.

### Clustered Index

A clustered index doesn't just point to your data — it **is** your data. The table rows are physically sorted on disk by the clustered index key. Think of a dictionary: the words are arranged alphabetically, and the definitions are right there with them. There's no separate lookup step.

```
Clustered Index (data sorted by primary key):
┌──────────────────────────────────────────────────┐
│ B+ Tree leaf pages ARE the table data             │
│                                                  │
│ Page 1: [id=1, Alice] [id=2, Bob] [id=3, Carol] │
│ Page 2: [id=4, Dave] [id=5, Eve] [id=6, Frank]  │
│ Page 3: [id=7, Grace] [id=8, Helen] ...         │
│                                                  │
│ The data IS the index — no extra lookup needed   │
└──────────────────────────────────────────────────┘
```

Because data can only be physically sorted one way, **you can only have one clustered index per table**. In MySQL InnoDB, this is always the primary key. In PostgreSQL, tables are stored as unsorted heaps by default (no clustered index).

Range queries on the clustered key are extremely fast because matching rows sit on consecutive disk pages — the database reads them sequentially.

### Non-Clustered (Secondary) Index

A non-clustered index is a **separate structure** that points back to the table data. Think of the index at the back of a book — it tells you "B-Trees: page 142" but the actual content is elsewhere. You look up the term, then flip to the page.

```
Non-Clustered Index (separate structure):
┌───────────────────────────┐      ┌──────────────────────┐
│ Index: users.email        │      │ Table Data           │
│                           │      │                      │
│ alice@... → PK=1  ────────┼──→   │ [id=1, Alice, ...]   │
│ bob@...   → PK=2  ────────┼──→   │ [id=2, Bob, ...]     │
│ carol@... → PK=3  ────────┼──→   │ [id=3, Carol, ...]   │
└───────────────────────────┘      └──────────────────────┘
                                    ↑ Extra hop to fetch row data
```

You can have **many** non-clustered indexes on a table — one for email, one for created_at, one for status, etc. But each one requires that extra hop to fetch the actual row, which means random I/O.

### How Different Databases Handle This

- **MySQL InnoDB**: The primary key is always the clustered index. Secondary indexes store the primary key value as their pointer, which means a secondary index lookup requires a second traversal of the primary key index (a "double lookup").
- **PostgreSQL**: No clustered index by default. All indexes are secondary, pointing to a heap table. You can run `CLUSTER` to sort once, but it's not maintained automatically.
- **SQL Server**: You choose which index to cluster on (defaults to primary key). Non-clustered indexes store a "row locator."
- **MongoDB**: The `_id` field acts as a clustered index in WiredTiger. Secondary indexes point to `_id`.

---

## Other Index Categories

Beyond clustered vs non-clustered, there are two more useful distinctions:

**Primary vs Secondary**: A primary index is automatically created on the primary key — it uniquely identifies each row. Secondary indexes are ones you create manually on other columns to speed up specific queries.

**Dense vs Sparse**: A dense index has one entry per row. A sparse index has one entry per page (or block). Sparse indexes are much smaller but only work when data is sorted (clustered), because knowing "key 150 is on Page 2" only helps if you can scan Page 2 to find the exact row.

```
Dense:  [1→Row1] [2→Row2] [3→Row3] [4→Row4] [5→Row5] ...

Sparse: [1→Page1] [100→Page2] [200→Page3] ...
        Page 1 contains rows 1-99 — scan within to find exact row
```

---

=== "Index Types"

    Now let's look at the actual data structures databases use to build indexes. Each one is optimized for a different access pattern.

    | Index Type | Lookup | Range Queries | Write Speed | Best For |
    |---|---|---|---|---|
    | B+ Tree | O(log n) | Excellent | Moderate | General purpose (default) |
    | Hash | O(1) | Not supported | Moderate | Exact-match lookups only |
    | LSM Tree | O(log n) | Good | Excellent | Write-heavy workloads |
    | Bitmap | Fast (bitwise) | Via bitwise OR | Slow (update all bitmaps) | OLAP, low-cardinality columns |
    | R-Tree / Spatial | O(log n) | Spatial regions | Moderate | Geographic / location data |
    | Inverted | O(1) per term | N/A | Moderate | Full-text search |

    ### B+ Tree -- The Workhorse

    The B+ Tree is the **default index structure in virtually every relational database** -- MySQL, PostgreSQL, Oracle, SQL Server, SQLite. Understanding it is non-negotiable for anyone working with databases.

    #### Why not just use a binary search tree?

    A binary search tree with 1 million keys has a depth of about 20. Each level requires one disk read. That's 20 disk reads per lookup, which at ~5ms each means 100ms per query.

    The insight behind B-Trees (and their refined cousin, the B+ Tree) is: **instead of 2 children per node, use hundreds or thousands.** Each node is sized to fit exactly one disk page (4-16 KB), so one disk read gives you hundreds of keys to compare against. A tree with branching factor 500 needs only 3 levels to hold 125 million keys -- that's 3 disk reads instead of 20.

    #### How B+ Tree Works

    The B+ Tree has two key design choices that make it superior for databases:

    1. **All data lives in leaf nodes only** -- internal nodes are purely for navigation
    2. **Leaf nodes are linked together** -- forming a chain for efficient range scans

    ```
    B+ Tree Structure:
                          ┌─────────────┐
                          │   30 | 60   │          ← Internal nodes: routing only
                          └──┬─────┬────┘            (no row data here)
                     ┌───────┘     └────────┐
                ┌────┴────┐           ┌──────┴──────┐
                │10|20|30 │←────────→│40|50|60|70  │  ← Leaf nodes: actual data
                │ ↓  ↓  ↓ │          │ ↓  ↓  ↓  ↓  │    + linked list for scans
                │Row Row Row│          │Row Row Row Row│
                └─────────┘          └─────────────┘
    ```

    **Point lookup** -- say you're searching for key 50:

    1. Read root node `[30 | 60]`: 50 > 30 and 50 < 60 → follow the middle pointer
    2. Read leaf node `[40|50|60|70]`: found 50 → return the row

    Two disk reads. The root is almost certainly cached in memory, so effectively one disk read.

    **Range query** -- this is where B+ Tree truly shines. Say you want `WHERE id BETWEEN 25 AND 55`:

    1. Traverse the tree to find the first leaf containing a key >= 25
    2. Follow the linked list across leaf nodes, collecting matching keys: 30, 40, 50
    3. Stop when you hit a key > 55

    This is all sequential I/O -- reading consecutive pages. On an HDD, sequential reads are 100x faster than random reads. On an SSD, it's still 10x faster.

    **How much data fits?** With MySQL InnoDB's 16 KB pages and 8-byte keys, each internal node holds about 1,170 entries. Three levels of the tree can index ~137 million rows. Four levels: ~160 billion rows. That's 3-4 disk reads to find any row in a table of virtually any size.

    #### Why not plain B-Tree?

    The original B-Tree stores data in both internal and leaf nodes. This means internal nodes are larger (they hold data alongside keys), which reduces the branching factor and makes the tree taller. It also makes range scans painful -- you have to zigzag up and down the tree to visit every key.

    The B+ Tree fixes both problems. By keeping internal nodes small (keys only), the branching factor is higher and the tree is shallower. And the leaf linked list makes range scans trivial. This is why every modern RDBMS uses B+ Tree, not plain B-Tree.

    B-Trees are still used in some file systems (NTFS, HFS+) and NoSQL stores where point queries dominate.

    ### Hash Index -- O(1) But Limited

    A hash index takes a fundamentally different approach: instead of maintaining a sorted tree, it runs each key through a **hash function** that maps directly to a storage location (bucket).

    ```
    Hash Index on users.email:

    "alice@example.com"
       ↓
    hash("alice@...") = 0x7A3F → bucket 319
       ↓
    ┌─────────────────────────────────────────┐
    │ Bucket 319:                             │
    │   "alice@example.com" → (Page 42, Row 3)│
    │   "xavier@corp.com"   → (Page 88, Row 12)│  ← collision
    └─────────────────────────────────────────┘

    Lookup: hash → bucket → scan 1-2 entries → done. O(1).
    ```

    The appeal is obvious: O(1) average lookup, compared to O(log n) for B+ Tree. For exact-match queries like `WHERE email = 'alice@example.com'`, hash is theoretically the fastest.

    **So why isn't hash the default?** Because it can *only* do exact matches. It fundamentally cannot support:

    - Range queries (`WHERE age > 25`) -- hash values have no ordering
    - Sorting (`ORDER BY name`) -- no sorted structure
    - Prefix matching (`WHERE name LIKE 'Al%'`) -- hash of "Alice" tells you nothing about "Al"
    - MIN/MAX -- no concept of "smallest" or "largest"

    Since most real-world workloads need at least some of these operations, B+ Tree wins as the default. Hash indexes are used in specific niches: Redis and Memcached (pure key-value lookups), MySQL's MEMORY engine, and PostgreSQL when you explicitly create one for equality-only queries on very long keys.

    When two keys hash to the same bucket (a **collision**), the most common strategies are chaining (linked list per bucket) or open addressing (probe the next slot). Performance degrades as the table fills up -- when load factor exceeds ~0.7, the entire hash table needs to be resized and every key rehashed.

    ### LSM Tree -- Built for Writes

    Every index type we've seen so far optimizes for reads. The **Log-Structured Merge Tree** flips the equation: it's designed for workloads that are overwhelmingly write-heavy.

    The core insight: disk **sequential** writes are 100-1000x faster than **random** writes. B+ Trees modify pages in place (random I/O). LSM Trees never modify anything in place -- all writes are sequential.

    #### The Write Path

    ```
    1. Write arrives
          ↓
    2. Append to Write-Ahead Log (sequential write, for crash recovery)
          ↓
    3. Insert into MemTable (in-memory sorted structure, like a red-black tree)
          ↓  (when MemTable fills up, ~64 MB)
    4. Flush to disk as an SSTable (Sorted String Table) — one big sequential write
          ↓  (background process)
    5. Compaction: merge smaller SSTables into larger ones, removing duplicates
    ```

    Because every write goes to an in-memory buffer first and flushes as a single sequential write, LSM Trees can absorb writes at extraordinary throughput.

    #### The Read Path (The Trade-off)

    Reading is where LSM Trees pay the price. To find a key, you must check:

    1. The current MemTable (in memory)
    2. Recently flushed SSTables (Level 0)
    3. Older, larger SSTables (Level 1, 2, 3...)

    That's potentially many files to check. **Bloom filters** are critical here -- they're probabilistic structures that can tell you "this file definitely does NOT contain your key" (with ~1% false positive rate), letting you skip most SSTables without reading them.

    #### When to Choose LSM over B+ Tree

    LSM Trees power Cassandra, RocksDB, LevelDB, HBase, ScyllaDB, and the storage engines behind CockroachDB and TiDB. Use them when:

    - Your workload is 90%+ writes (event logging, metrics, IoT sensors, time-series)
    - You need append-mostly patterns (audit logs, activity feeds)
    - You can tolerate slightly slower reads

    Avoid them when reads must be consistently fast (LSM compaction can cause latency spikes) or when you're space-constrained (compaction temporarily needs extra disk space).

    ### Bitmap Index -- For Analytics

    Bitmap indexes take a completely different approach suited to **analytical queries on low-cardinality columns** (columns with few distinct values like status, department, or country).

    For each distinct value, a bitmap index maintains a **bit array** where bit `i` is 1 if row `i` has that value.

    ```
    Table: employees (8 rows)
    ┌────┬──────┬────────┐
    │ ID │ Dept │ Status │
    ├────┼──────┼────────┤
    │ 0  │ Eng  │ Active │
    │ 1  │ Sales│ Active │
    │ 2  │ Eng  │ Left   │
    │ 3  │ HR   │ Active │
    │ 4  │ Eng  │ Active │
    │ 5  │ Sales│ Left   │
    │ 6  │ HR   │ Active │
    │ 7  │ Eng  │ Active │
    └────┴──────┴────────┘

    Bitmap for Dept:              Bitmap for Status:
      Eng:   [1,0,1,0,1,0,0,1]     Active: [1,1,0,1,1,0,1,1]
      Sales: [0,1,0,0,0,1,0,0]     Left:   [0,0,1,0,0,1,0,0]
      HR:    [0,0,0,1,0,0,1,0]
    ```

    The magic happens with **bitwise operations**. "Find active engineers" becomes:

    ```
      Eng    = [1,0,1,0,1,0,0,1]
      Active = [1,1,0,1,1,0,1,1]
      AND    = [1,0,0,0,1,0,0,1]  → Rows 0, 4, 7
    ```

    This is a single CPU instruction operating on the entire dataset. For complex multi-column filters in analytical queries, bitmaps are extraordinarily fast.

    The catch: bitmap indexes are terrible for writes (every insert may require updating multiple bitmaps) and terrible for high-cardinality columns (one bitmap per distinct value becomes enormous). They shine in OLAP systems like Oracle's analytics engine, Apache Druid, and ClickHouse.

    ### Spatial Index (R-Tree) -- For Geographic Data

    B+ Trees work because you can sort data in one dimension. But how do you sort latitude/longitude pairs? You can't -- there's no single ordering that preserves proximity in 2D space.

    **R-Trees** solve this by grouping nearby objects into nested **minimum bounding rectangles (MBRs)**.

    ```
    R-Tree for restaurant locations:

    ┌──────────────────────────────────────────┐
    │               City (Root MBR)            │
    │   ┌────────────────┐                     │
    │   │ MBR A (north)  │                     │
    │   │  •R1   •R2     │                     │
    │   │     •R3        │                     │
    │   └────────────────┘                     │
    │              ┌─────────────────┐         │
    │              │ MBR B (south)   │         │
    │              │   •R4      •R5  │         │
    │              │        •R6      │         │
    │              └─────────────────┘         │
    └──────────────────────────────────────────┘

    Query: "Restaurants within 2km of me"
      Does search circle overlap MBR A? → Yes → check R1, R2, R3
      Does search circle overlap MBR B? → No → skip entirely!
      Only checked 3 restaurants instead of 6.
    ```

    Other spatial indexing approaches include **Geohash** (converts 2D coordinates into a 1D string), **Quad-Trees** (recursively divide space into four quadrants), and **H3** (Uber's hexagonal grid system for ride matching).

    Real-world usage: PostGIS uses GiST (generalized R-Tree) for geographic queries, MongoDB has `2dsphere` indexes, Google Maps uses S2 cells, and Uber uses H3 hexagonal grids for surge pricing zones.

    ### Inverted Index -- For Full-Text Search

    Normal indexes map **row → data**. An inverted index flips this: it maps **word → list of rows containing that word**. This inversion makes full-text search possible.

    ```
    Documents:
      Doc 1: "database indexing improves query performance"
      Doc 2: "database sharding scales write throughput"
      Doc 3: "indexing and sharding are core database concepts"

    After tokenizing and stemming:
    ┌──────────────┬──────────────────┐
    │ Term         │ Posting List     │
    ├──────────────┼──────────────────┤
    │ database     │ [Doc1, Doc2, Doc3]│
    │ index        │ [Doc1, Doc3]     │
    │ shard        │ [Doc2, Doc3]     │
    │ query        │ [Doc1]           │
    │ throughput   │ [Doc2]           │
    │ ...          │                  │
    └──────────────┴──────────────────┘

    Search: "database indexing"
      "database" → [Doc1, Doc2, Doc3]
      "index"    → [Doc1, Doc3]
      Intersect  → [Doc1, Doc3]
    ```

    Real search engines enhance this with term frequency (for relevance scoring), positions (for phrase matching), and skip pointers (for faster intersection of long posting lists). The scoring algorithm **BM25** (used by Elasticsearch) ranks results by how rare and concentrated the search terms are in each document.

    Elasticsearch and Apache Solr (both built on Lucene) are the dominant full-text search systems. PostgreSQL offers built-in full-text search through GIN indexes with `tsvector`/`tsquery`. MySQL has `FULLTEXT` indexes. MongoDB has text indexes and Atlas Search (powered by Lucene).

=== "Advanced Indexing"

    ### Composite and Covering Indexes

    So far we've looked at indexes on a single column. In practice, many queries filter on multiple columns, and this is where **composite indexes** become essential.

    #### Composite (Multi-Column) Indexes

    A composite index is an index on multiple columns. The critical thing to understand is that **column order matters** -- the index sorts by the first column, then by the second within ties, then by the third. This gives rise to the **leftmost prefix rule**:

    ```
    Composite Index: (department, status, hire_date)

    The index can help with:
      WHERE department = 'Eng'
      WHERE department = 'Eng' AND status = 'Active'
      WHERE department = 'Eng' AND status = 'Active' AND hire_date > '2024-01-01'
      WHERE department = 'Eng' ORDER BY status

    The index CANNOT help with:
      WHERE status = 'Active'              — first column skipped
      WHERE hire_date > '2024-01-01'       — first two columns skipped
      WHERE department = 'Eng' AND hire_date > ?  — middle column skipped
    ```

    Why? Think of a phone book sorted by (last name, first name). You can look up everyone named "Smith" or "Smith, Alice," but you can't efficiently look up everyone named "Alice" -- the Alices are scattered across different last names.

    **Column ordering guidelines:**

    - Put equality columns (`=`) before range columns (`>`, `<`, `BETWEEN`)
    - Put high-selectivity columns (those that filter out more rows) first
    - Match the order to your most common query patterns

    #### Covering Indexes (Index-Only Scans)

    A covering index contains **all the columns** a query needs, so the database never has to touch the actual table at all.

    ```
    Query: SELECT email, status FROM users WHERE department = 'Eng'

    Without covering index:
      1. Scan index on (department) → get row pointers
      2. For EACH matching row, jump to the table to fetch email and status
      → Lots of random I/O

    With covering index on (department, email, status):
      1. Scan index → email and status are already in the index leaf nodes
      2. Return results directly
      → Zero table access! "Index-Only Scan" in EXPLAIN output.
    ```

    PostgreSQL and SQL Server support `INCLUDE` columns: `CREATE INDEX idx ON users(department) INCLUDE (email, status)`. This keeps email and status in the leaf nodes for covering purposes without affecting the sort order. MySQL achieves the same by including all needed columns directly in the composite index.

    ### Write Amplification: The Hidden Cost

    When a table has 5 indexes, every INSERT does 6 operations: 1 table write + 5 index updates. This is **write amplification**, and it's the reason you can't just "add an index for everything."

    ```
    INSERT INTO users (...) VALUES (...);

    What actually happens:
      1. Write row to table                  ← 1 write
      2. Update primary key index (B+ Tree)  ← 1 write
      3. Update email index (B+ Tree)        ← 1 write
      4. Update (dept, status) index         ← 1 write
      5. Update created_at index             ← 1 write
      6. Update full-text index              ← N writes (one per word)

    Total: 6+ writes for 1 logical INSERT
    ```

    Over time, indexes also accumulate **bloat** -- when rows are deleted or updated, the old index entries aren't immediately reclaimed. They become dead space that wastes memory and slows scans. PostgreSQL offers `REINDEX CONCURRENTLY` to rebuild without locking; MySQL uses `ALTER TABLE ... ENGINE=InnoDB` for online rebuilds.

=== "Database Specifics"

    ### MySQL InnoDB

    InnoDB's defining characteristic is that the **primary key is always the clustered index**. The table data is physically organized as a B+ Tree sorted by primary key. This makes primary key lookups extremely fast but has an important consequence: secondary indexes store the primary key value as their pointer, so a secondary index lookup requires two B+ Tree traversals (one through the secondary index, one through the primary key index to fetch the actual row).

    InnoDB also has some clever optimizations: an **adaptive hash index** that automatically builds an in-memory hash for frequently accessed B+ Tree pages, and a **change buffer** that batches secondary index updates to reduce random I/O.

    ### PostgreSQL

    PostgreSQL takes a different approach -- tables are stored as **unordered heaps**, and all indexes (even on the primary key) are secondary. This means every index lookup requires a separate table access. PostgreSQL compensates with its rich set of index types:

    - **B-Tree** (default) -- the standard sorted index for 90% of use cases
    - **Hash** -- for equality-only queries on large keys
    - **GIN** -- generalized inverted index for full-text search, JSONB, and arrays
    - **GiST** -- generalized search tree for spatial data (PostGIS), range types
    - **BRIN** -- block range index, PostgreSQL's secret weapon for huge, naturally-ordered tables

    **BRIN deserves special mention.** For a billion-row events table where data is inserted roughly in time order, a B-Tree index on timestamp might be 20 GB. A BRIN index stores only the min/max timestamp per block range -- it might be 1 MB. The trade-off: BRIN only works when data is physically ordered (high correlation between row position and column value).

    ```
    BRIN on events.timestamp (1 billion rows):

    Block Range 1-128:   min=2024-01-01, max=2024-01-15 → skip
    Block Range 129-256: min=2024-01-15, max=2024-01-30 → skip
    ...
    Block Range N:       min=2024-06-01, max=2024-06-15 → scan this range

    20,000x smaller than a B-Tree index.
    ```

    ### MongoDB

    MongoDB's WiredTiger engine uses B-Trees for all indexes. The `_id` field acts as a clustered index. MongoDB supports specialized index types including geospatial (`2dsphere`), text indexes for full-text search, hashed indexes for even shard distribution, and wildcard indexes for dynamic schemas.

    ### Choosing the Right Index

    When faced with a new query to optimize, this decision flow will guide you:

    ```
    What's your query pattern?
        │
        ├─ Exact match only (WHERE col = value)
        │   ├─ Ultra-high throughput, never needs range/sort? → Hash Index
        │   └─ General use? → B+ Tree (handles everything)
        │
        ├─ Range queries (>, <, BETWEEN, ORDER BY)
        │   └─ B+ Tree
        │
        ├─ Full-text search (keywords, relevance ranking)
        │   └─ Inverted Index (GIN in PG, FULLTEXT in MySQL)
        │
        ├─ Geographic / spatial (nearby, within radius)
        │   └─ R-Tree / GiST / Geohash
        │
        ├─ Low-cardinality + complex filters (OLAP/analytics)
        │   └─ Bitmap Index
        │
        ├─ Write-heavy workload (90%+ writes)
        │   └─ LSM Tree (Cassandra, RocksDB)
        │
        └─ Huge naturally-ordered table, rare updates
            └─ BRIN (PostgreSQL)
    ```

    For composite indexes, remember: **equality columns first, range columns last**, and always check with `EXPLAIN ANALYZE` that the database is actually using your index.

---

## Common Mistakes

**Over-indexing.** Adding 10+ indexes on an OLTP table grinds writes to a halt. Aim for 5-7 indexes maximum on write-heavy tables. Monitor unused indexes (check `pg_stat_user_indexes` in PostgreSQL where `idx_scan = 0`) and drop them — they're pure write overhead.

**Wrong column order in composites.** An index on `(status, user_id)` doesn't help queries that filter by `user_id` first. Column order must match your query patterns, with high-selectivity and equality columns first.

**Functions that break index usage.** `WHERE UPPER(email) = 'BOB@EXAMPLE.COM'` can't use an index on `email` because the database sees a function call, not a column reference. Fix: create an expression index on `UPPER(email)`.

**Duplicate indexes.** Having both `(A)` and `(A, B)` is redundant — the composite `(A, B)` already handles queries on just `A`.

**Never checking EXPLAIN.** Creating an index doesn't guarantee the query planner will use it. Always verify with `EXPLAIN` (or `EXPLAIN ANALYZE` for actual execution stats).

**Indexing tiny tables.** For tables under ~1,000 rows, a full table scan is often faster than an index lookup because the entire table fits in a couple of pages. The database's query planner usually knows this and ignores the index anyway.

---

## Key Takeaways

1. **B+ Tree is the default for a reason** — it handles equality, ranges, sorting, and prefix matching. Start here unless you have a specific reason not to.

2. **Clustered index = physical order** — you get one per table. In MySQL it's always the primary key; choose it wisely.

3. **Every index slows writes** — it's a conscious trade-off. Don't add indexes speculatively; add them when you see slow queries.

4. **Column order in composites is crucial** — the leftmost prefix rule determines which queries can use the index.

5. **Covering indexes avoid table lookups entirely** — include frequently selected columns to enable index-only scans.

6. **LSM Trees trade read speed for write throughput** — use them (Cassandra, RocksDB) for write-heavy workloads.

7. **Always verify with EXPLAIN** — creating an index is only half the job; confirming the planner uses it is the other half.

---

## Related Topics

- **[Sharding](sharding.md)** — indexes work within each shard; shard key selection relates to indexing strategy
- **[Database Types](database-types.md)** — index types and capabilities vary by database model
