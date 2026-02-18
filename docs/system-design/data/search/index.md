# Search and Indexing Systems

Your database can answer "give me the row where id = 42" in milliseconds. But ask it "find all products matching 'blue running shoes' sorted by relevance" and things fall apart. Regular databases are optimized for exact lookups and structured queries — they index by column values, not by the meaning of text. Search is a fundamentally different problem that requires fundamentally different data structures.

This is why every major platform — Google, Amazon, Twitter, LinkedIn — runs a dedicated search infrastructure alongside its databases. The database is the system of record; the search engine is the system of discovery.

---

## High-Level Architecture

In most production systems, search lives as a separate service that mirrors a subset of data from the primary database. This separation exists because the data structures that make search fast (inverted indexes) are very different from the structures that make transactions safe (B-trees, WAL logs).

```
                         ┌──────────────┐
                         │   Client     │
                         └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    v                       v
            ┌──────────────┐       ┌──────────────┐
            │  Application │       │  Application │
            │   Server     │       │   Server     │
            └──────┬───────┘       └──────┬───────┘
                   │                      │
          ┌────────┴────────┐             │
          │                 │             │
          v                 v             v
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  Database   │  │   Search    │  │   Search    │
   │  (Primary)  │──│   Indexer   │──│   Cluster   │
   │  PostgreSQL │  │   (CDC)     │  │ Elasticsearch│
   └─────────────┘  └─────────────┘  └─────────────┘
    Source of truth   Keeps search     Handles queries
    Writes go here    index in sync    Reads go here
```

**Why not just add full-text search to the database?** PostgreSQL has `tsvector`, MySQL has `FULLTEXT` indexes. These work for small-scale, simple search. But they break down when you need: relevance scoring across millions of documents, sub-100ms response times, faceted filtering, autocomplete, fuzzy matching, or horizontal scaling of search load independently from database load.

---

## The Inverted Index

The inverted index is the core data structure behind every search engine. A normal database index maps from a row ID to its content (forward index). An inverted index flips this: it maps from every word to the list of documents containing that word.

### How It Works

Consider three product descriptions being indexed:

```
Doc 1: "Blue running shoes for men"
Doc 2: "Red running shoes for women"
Doc 3: "Blue hiking boots waterproof"
```

**Step 1 — Tokenization:** Break each document into individual terms.

```
Doc 1 → [blue, running, shoes, for, men]
Doc 2 → [red, running, shoes, for, women]
Doc 3 → [blue, hiking, boots, waterproof]
```

**Step 2 — Normalization:** Lowercase, stem, remove stop words (details in next section).

```
Doc 1 → [blue, run, shoe, men]
Doc 2 → [red, run, shoe, women]
Doc 3 → [blue, hike, boot, waterproof]
```

**Step 3 — Build the inverted index:** For each term, record which documents contain it and where.

```
Inverted Index:
┌──────────────┬─────────────────────────────────┐
│    Term      │   Postings List                  │
├──────────────┼─────────────────────────────────┤
│ blue         │ Doc 1 (pos: 1), Doc 3 (pos: 1)  │
│ boot         │ Doc 3 (pos: 3)                   │
│ hike         │ Doc 3 (pos: 2)                   │
│ men          │ Doc 1 (pos: 4)                   │
│ red          │ Doc 2 (pos: 1)                   │
│ run          │ Doc 1 (pos: 2), Doc 2 (pos: 2)  │
│ shoe         │ Doc 1 (pos: 3), Doc 2 (pos: 3)  │
│ waterproof   │ Doc 3 (pos: 4)                   │
│ women        │ Doc 2 (pos: 4)                   │
└──────────────┴─────────────────────────────────┘
```

**Step 4 — Query resolution:** When a user searches "blue shoes," the engine looks up "blue" and "shoe" in the index, finds the intersection (Doc 1 appears in both), and returns Doc 1 as the top result. Doc 3 matches "blue" but not "shoe," so it ranks lower.

```
Query: "blue shoes"
  │
  ├── "blue"  → {Doc 1, Doc 3}
  ├── "shoe"  → {Doc 1, Doc 2}
  │
  └── Intersection: Doc 1 (matches both terms → highest relevance)
      Partial:     Doc 2 (matches "shoe" only)
      Partial:     Doc 3 (matches "blue" only)
```

### Why This Is Fast

A B-tree index in a database would need to scan every row looking for the word "blue" inside a text column — essentially a `LIKE '%blue%'` full table scan. The inverted index turns this into a direct dictionary lookup: go to the "blue" entry, get the document list. For a corpus of 100 million documents, this lookup takes microseconds, not minutes.

The postings lists are stored sorted by document ID, which makes set intersection (AND queries) and set union (OR queries) extremely efficient using merge algorithms.

---

## Text Processing Pipeline

Raw text must pass through several transformations before it reaches the inverted index. The quality of search results depends heavily on this pipeline — the same pipeline runs at index time (when documents are added) and at query time (when users search).

```
Raw Text
  │
  v
┌─────────────────┐
│  Tokenization   │  "New York's best-selling books" → [New, York's, best-selling, books]
└────────┬────────┘
         v
┌─────────────────┐
│ Lowercasing     │  [new, york's, best-selling, books]
└────────┬────────┘
         v
┌─────────────────┐
│ Stop Word       │  [new, york's, best-selling, books]  (no stop words here)
│ Removal         │  "the quick brown fox" → [quick, brown, fox]
└────────┬────────┘
         v
┌─────────────────┐
│ Stemming /      │  "running" → "run", "books" → "book"
│ Lemmatization   │  "better" → "good" (lemmatization only)
└────────┬────────┘
         v
┌─────────────────┐
│ Synonym         │  "NYC" → "new york city"
│ Expansion       │  "laptop" → "laptop, notebook"
└────────┬────────┘
         v
  Index-ready tokens
```

### Tokenization

Tokenization splits text into individual terms. This sounds simple until you consider:

- **Hyphenated words:** Is "best-selling" one token or two? Most analyzers split it into ["best", "selling"].
- **Possessives:** "York's" becomes "york" after removing the possessive.
- **Numbers:** "iPhone 14 Pro" — should "14" be searchable? Usually yes.
- **CJK languages:** Chinese, Japanese, and Korean don't use spaces between words. Tokenization requires dictionary-based segmentation.
- **Email and URLs:** "user@example.com" might tokenize as ["user", "example", "com"] or be kept whole, depending on configuration.

### Stemming vs. Lemmatization

Both reduce words to a root form so that "running," "runs," and "ran" all match the same index entry. The approaches differ in sophistication:

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Method** | Chops off suffixes using rules | Uses dictionary + grammar |
| **Speed** | Very fast (rule-based) | Slower (dictionary lookup) |
| **Accuracy** | Sometimes wrong | Linguistically correct |
| **Example** | "better" -> "better" (no rule) | "better" -> "good" |
| **Example** | "studies" -> "studi" | "studies" -> "study" |
| **Example** | "running" -> "run" | "running" -> "run" |
| **Used by** | Elasticsearch (default) | SpaCy, NLTK |

**Trade-off:** Stemming is good enough for most search use cases and is significantly faster. Lemmatization matters when precision is critical, such as in medical or legal search where "operating" (surgery) and "operating" (a business) have different meanings.

### Stop Words

Stop words are extremely common words ("the," "is," "at," "and") that appear in nearly every document. Since they don't help distinguish one document from another, many search engines remove them from the index.

**However, this is not always correct.** The query "to be or not to be" becomes empty after stop word removal. Modern search engines like Elasticsearch have moved away from aggressive stop word removal, instead relying on TF-IDF scoring to naturally down-weight common terms.

### Relevance Scoring: TF-IDF and BM25

After finding documents that match the query terms, the search engine must rank them. Not all matches are equally relevant. The two foundational scoring algorithms:

**TF-IDF (Term Frequency - Inverse Document Frequency)**

The intuition: a term is important to a document if it appears frequently in that document (TF) but rarely across all documents (IDF).

```
TF  = (times term appears in document) / (total terms in document)
IDF = log(total documents / documents containing term)

Score = TF x IDF
```

- "the" has high TF in every document but very low IDF (appears everywhere) -> low score
- "elasticsearch" has moderate TF in a technical article and high IDF (rare word) -> high score

**BM25 (Best Match 25)**

BM25 is the evolution of TF-IDF and is the default scoring algorithm in Elasticsearch and Solr. Two key improvements:

- **Term frequency saturation:** In TF-IDF, if a word appears 100 times, the score is 10x higher than if it appears 10 times. BM25 applies diminishing returns — after a certain point, more occurrences barely increase the score. A document mentioning "shoes" 50 times is not 50x more relevant than one mentioning it once.
- **Document length normalization:** A 10,000-word article naturally contains more term occurrences than a 100-word product description. BM25 normalizes for this, so short, focused documents are not penalized against long ones.

```
BM25 scoring concept:

Relevance
  ^
  │        ╭────────────────────  BM25 (saturates)
  │       ╱
  │      ╱          ╱  TF-IDF (grows linearly)
  │     ╱          ╱
  │    ╱         ╱
  │   ╱        ╱
  │  ╱       ╱
  │ ╱      ╱
  │╱     ╱
  └──────────────────────────> Term Frequency
```

In practice, BM25 produces better results for nearly all use cases. You rarely need to understand the full math — just know that it rewards relevant terms, punishes overly common terms, and handles documents of different lengths gracefully.

---

## Search Architecture Patterns

### Pattern 1: Search as a Separate Service

The most common production pattern. The search engine runs as an independent cluster, completely decoupled from the primary database.

```
┌─────────────┐     writes      ┌─────────────┐
│ Application ├────────────────>│  Database   │
│   Server    │                 │ (PostgreSQL)│
│             │   search query  └──────┬──────┘
│             ├──────────────┐         │
└─────────────┘              │    CDC / sync
                             v         │
                    ┌──────────────┐   │
                    │    Search    │<──┘
                    │   Cluster   │
                    │(Elasticsearch)│
                    └──────────────┘
```

**How it works:** Writes go to the database. A synchronization process (CDC, message queue, or periodic batch) pushes changes to the search cluster. Search queries go directly to the search cluster, never touching the database.

**Advantages:** Search scales independently, search downtime does not affect writes, you can tune search hardware separately.

**Disadvantage:** Data in search is eventually consistent with the database — there is always some lag.

### Pattern 2: Near-Real-Time Indexing via CDC

Change Data Capture (CDC) reads the database transaction log and streams every insert, update, and delete to the search engine. This is how most large-scale systems keep search in sync.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Database │───>│  CDC     │───>│  Message │───>│   Search     │
│  (Write) │    │ (Debezium│    │  Queue   │    │   Indexer    │
│          │    │  / WAL)  │    │ (Kafka)  │    │              │
└──────────┘    └──────────┘    └──────────┘    └──────┬───────┘
                                                       │
                                                       v
                                                ┌──────────────┐
                                                │   Search     │
                                                │   Cluster    │
                                                └──────────────┘
```

**Typical latency:** 1-5 seconds from database write to searchable. Twitter uses a similar pipeline to make tweets searchable within seconds of posting.

**Why Kafka in the middle?** The message queue absorbs spikes. If the search indexer goes down for maintenance, events queue up in Kafka and get processed when it comes back — no data loss.

**Real-world example:** LinkedIn uses CDC from their primary database through Kafka to power search across 900M+ profiles. When a user updates their job title, it becomes searchable within seconds.

### Pattern 3: Sharded Search

When a single search node cannot hold the entire index in memory, you must shard the search cluster. There are two strategies:

**Document-partitioned (most common):** Each shard holds a complete inverted index for a subset of documents.

```
Document-Partitioned Search:

Query: "blue shoes"
        │
        v
  ┌───────────┐
  │  Search   │  Scatter query to all shards
  │  Router   │
  └─────┬─────┘
    ┌───┼───────────┐
    v   v           v
┌───────┐ ┌───────┐ ┌───────┐
│Shard 1│ │Shard 2│ │Shard 3│
│Doc 1-3│ │Doc 4-6│ │Doc 7-9│  Each shard searches
│million│ │million│ │million│  its own documents
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              v
        Merge & Rank        Gather results, re-rank
        top-K results       globally, return top N
```

**Term-partitioned (specialized):** Each shard owns specific terms from the vocabulary. A query for "blue shoes" goes to the shard owning "blue" and the shard owning "shoes," then results are intersected. This avoids scatter-gather but makes multi-term queries complex.

| Aspect | Document-Partitioned | Term-Partitioned |
|--------|---------------------|------------------|
| **Query routing** | Every query hits all shards | Query goes to specific shards |
| **Load balancing** | Naturally balanced | Hot terms create hot shards |
| **Scaling** | Add shards, redistribute docs | Must rebalance vocabulary |
| **Used by** | Elasticsearch, Solr | Google (hybrid approach) |
| **Best for** | General-purpose search | Extremely large vocabularies |

In practice, document-partitioned is the standard. Elasticsearch and Solr both use it exclusively.

---

## Technology Comparison

| Feature | Elasticsearch | Solr | Meilisearch | Typesense |
|---------|--------------|------|-------------|-----------|
| **Built on** | Lucene | Lucene | Rust (custom) | C++ (custom) |
| **Primary use** | General search + analytics | Enterprise search | Instant search (typo-tolerant) | Instant search (simple) |
| **Cluster scaling** | Excellent (auto-sharding) | Good (manual config) | Limited (single node focus) | Good (built-in clustering) |
| **Query language** | JSON DSL (powerful, complex) | Solr Query Parser / JSON | Simple key-value params | Simple key-value params |
| **Relevance tuning** | Extensive (function_score, etc.) | Extensive (similar to ES) | Limited but sensible defaults | Moderate |
| **Typo tolerance** | Via fuzzy queries (manual) | Via fuzzy queries (manual) | Built-in, out of the box | Built-in, out of the box |
| **Setup complexity** | High (JVM tuning, config) | High (JVM tuning, config) | Very low (single binary) | Very low (single binary) |
| **Memory usage** | Heavy (JVM, ~4-8 GB min) | Heavy (JVM, ~4-8 GB min) | Light (~100 MB) | Light (~100 MB) |
| **Best for** | Large-scale, complex search | Enterprise with existing Java stack | Frontend instant search, prototypes | Simple search, small-medium data |
| **Scale** | Billions of documents | Billions of documents | Millions of documents | Millions of documents |
| **License** | SSPL (not truly open) | Apache 2.0 | MIT | GPL-3.0 |

### When to Choose What

- **Elasticsearch:** Your default choice for production search at scale. Massive ecosystem, battle-tested at companies like Netflix, Uber, and Wikipedia. Choose when you need complex queries, aggregations, or analytics alongside search.
- **Solr:** If you already have a Java ecosystem and need similar power to Elasticsearch. Solr is more stable for certain enterprise use cases but has a smaller community.
- **Meilisearch:** Perfect for frontend instant search where typo tolerance matters (search-as-you-type). Dramatically simpler to operate than Elasticsearch. Choose for small-to-medium datasets (under 10M documents).
- **Typesense:** Similar niche to Meilisearch. Slightly better clustering support. Choose when you need a lightweight, easy-to-operate search engine with good typo tolerance.

---

## Real-World Examples

### Google: Search at Planetary Scale

Google's search system processes 8.5 billion queries per day across an index of hundreds of billions of web pages.

**Architecture highlights:**

- The web is crawled continuously by Googlebot, which discovers and fetches pages
- Documents pass through a massive text processing pipeline (tokenization, language detection, entity extraction, quality scoring)
- The inverted index is sharded across thousands of servers in multiple data centers
- PageRank adds a layer beyond text relevance: a page linked to by many authoritative sites ranks higher, even if its text match is weaker
- Multiple index tiers: a "hot" tier for frequently accessed pages (served from memory), a "cold" tier for the long tail

```
Google Search (simplified):
┌─────────┐    ┌──────────────┐    ┌──────────────┐
│Googlebot│───>│ Index Build  │───>│  Serving     │
│ (Crawl) │    │ Pipeline     │    │  System      │
└─────────┘    └──────────────┘    └──────┬───────┘
                                          │
                    ┌─────────────────────┼────────────────────┐
                    │                     │                    │
              ┌─────┴─────┐        ┌─────┴─────┐       ┌─────┴─────┐
              │  Index    │        │  Index    │       │  Index    │
              │  Shard 1  │        │  Shard 2  │       │  Shard N  │
              │  (hot)    │        │  (hot)    │       │  (cold)   │
              └───────────┘        └───────────┘       └───────────┘

Query flow: User query → spell check → query expansion →
            scatter to shards → merge results →
            re-rank with PageRank + ML → return top 10
```

**Key insight:** Google's ranking is not purely text-based. Hundreds of signals (PageRank, freshness, user location, click-through rates, page speed) combine through machine learning models to determine the final ranking. Text relevance is just one input.

### Twitter: Real-Time Search at 500M Tweets/Day

Twitter's search system must index approximately 500 million tweets per day and make them searchable within seconds.

**Architecture highlights:**

- Tweets flow through Kafka into a real-time indexing pipeline called Earlybird
- Earlybird maintains an in-memory inverted index for recent tweets (last 7 days)
- Older tweets move to an on-disk index cluster for historical search
- Search results are blended from real-time and historical clusters, then ranked by recency, engagement, and relevance

```
Tweet lifecycle:
  Tweet posted → Kafka → Earlybird (in-memory, <10 sec latency)
                              │
                              │ (after 7 days)
                              v
                    Archive Index (on-disk)
```

**Key challenge:** At Twitter's scale, the "fan-out" problem is real. A trending topic might match millions of tweets. The system must return relevant results in under 200ms despite this massive result set. Twitter uses aggressive top-K pruning — once enough high-quality results are found, scanning stops.

### LinkedIn: Searching 900M+ Profiles

LinkedIn's search spans multiple entity types: people, jobs, companies, posts, and groups. Each entity type has its own search vertical.

**Architecture highlights:**

- Galene, LinkedIn's custom search engine, handles federated search across entity types
- Profile updates flow through CDC (Kafka) to keep the search index fresh
- Search is personalized: when you search "software engineer," the results are ranked by your network proximity, shared connections, and your industry
- Query understanding layer: "SWE at FAANG" is expanded to "Software Engineer at Facebook, Amazon, Apple, Netflix, Google"

**Scale numbers:** 900M+ member profiles, each with hundreds of fields. The search cluster processes tens of thousands of queries per second, with p99 latency under 200ms.

### E-commerce: Faceted Search

E-commerce search adds a layer beyond text matching: faceted filtering. When you search "laptop" on Amazon, you see results alongside filter panels for brand, price range, screen size, RAM, customer rating.

```
E-commerce search flow:

User query: "laptop under $1000"
  │
  v
┌────────────────────────────────────────────┐
│             Search Results                 │
│                                            │
│  Filters (facets):     │  Results:         │
│  ┌──────────────────┐  │  1. Dell XPS 13  │
│  │ Brand            │  │     $899 ★★★★☆   │
│  │ □ Dell    (42)   │  │  2. MacBook Air  │
│  │ □ Apple   (38)   │  │     $999 ★★★★★   │
│  │ □ Lenovo  (35)   │  │  3. ThinkPad T14 │
│  │ ├─────────────── │  │     $849 ★★★★☆   │
│  │ Price Range      │  │                   │
│  │ □ $500-$750 (28) │  │                   │
│  │ □ $750-$1000(45) │  │                   │
│  │ ├─────────────── │  │                   │
│  │ Screen Size      │  │                   │
│  │ □ 13" (30)       │  │                   │
│  │ □ 14" (25)       │  │                   │
│  │ □ 15" (22)       │  │                   │
│  └──────────────────┘  │                   │
└────────────────────────────────────────────┘
```

**How facets work:** When Elasticsearch executes a search, it can simultaneously compute aggregations — counting how many results fall into each category. The counts next to each filter ("Dell (42)") are computed in real-time from the matching document set, not pre-calculated.

**Challenge:** Faceted search is expensive. Each facet is essentially a GROUP BY across the result set. Sites like Amazon may have 20+ facets per category, each requiring an aggregation pass. Caching popular facet combinations and limiting facet depth are common optimizations.

---

## Advanced Search Topics

### Autocomplete (Search-as-You-Type)

Autocomplete must return suggestions within 50-100ms as the user types each character. Three common approaches:

- **Prefix matching on an edge n-gram index:** At index time, "elasticsearch" is stored as ["e", "el", "ela", "elas", ...]. A prefix query matches instantly against these pre-computed prefixes. Fast but increases index size.
- **Completion suggester (Elasticsearch):** A specialized in-memory data structure (FST — Finite State Transducer) optimized for prefix lookups. Very fast but limited to exact prefix matches.
- **Trie-based suggestions:** A separate trie data structure stores popular queries. As the user types, the trie is traversed to find completions. Used by Google for query suggestions.

### Fuzzy Matching (Typo Tolerance)

Users misspell things. "restraunt" should still find "restaurant." Fuzzy matching uses edit distance (Levenshtein distance) to find terms within N character changes of the query.

```
Edit distance examples:
  "kitten" → "sitting"  = 3 edits (k→s, e→i, +g)
  "restraunt" → "restaurant" = 2 edits (swap letters)

Typical config: allow 1-2 edits for terms > 5 characters
```

**Trade-off:** Fuzzy matching is expensive — it must check many candidate terms. Most engines limit fuzziness to 1-2 edits and only for longer terms (short words like "cat" would match too many things with 2 edits).

### Geo-Search

Finding results near a location (restaurants within 5km, stores in a city) requires geospatial indexing. Search engines use structures like geohashes or R-trees to partition space.

```
Geohash grid:
┌─────┬─────┬─────┐
│ 9q8 │ 9q9 │ 9qd │
├─────┼─────┼─────┤   "Restaurants near me"
│ 9q2 │ 9q3*│ 9q6 │   * = user location
├─────┼─────┼─────┤   Search cells 9q3 + adjacent
│ 9q0 │ 9q1 │ 9q4 │
└─────┴─────┴─────┘
```

The search engine finds all documents whose geohash shares a prefix with the user's location. Longer shared prefixes mean closer proximity. Combined with text search: "pizza near 9q3" searches the inverted index for "pizza" and filters by geohash proximity.

### Vector Search (Semantic Search)

Traditional search matches keywords. Vector search matches meaning. Text is converted into high-dimensional vectors using ML models (embeddings), and search finds the nearest vectors to the query.

```
Keyword search:  "affordable sedan"  → matches documents containing these words
Vector search:   "affordable sedan"  → finds "budget-friendly car", "cheap automobile"
                                       (same meaning, different words)
```

This is the foundation of modern AI-powered search. Elasticsearch 8.x, Pinecone, Weaviate, and Milvus all support vector search. Many production systems use hybrid search: combine keyword matching (BM25) with vector similarity for the best results.

---

## Common Challenges

### 1. Index Consistency

The search index is a derived copy of the database. They can drift apart:

- **Failure during sync:** The CDC pipeline drops an event. A document is updated in the database but the search index still has the old version.
- **Ordering issues:** Two rapid updates to the same document arrive at the search indexer out of order. The older version overwrites the newer one.
- **Schema changes:** A new field is added to the database but not mapped in the search index.

**Mitigations:** Include version numbers or timestamps in every document. The indexer rejects updates with older versions. Run periodic reconciliation jobs that compare database records against search index records and fix discrepancies.

### 2. Search Relevance Tuning

Getting relevance right is an ongoing process, not a one-time configuration. Common issues:

- **Exact match vs. broad match:** A search for "python" should surface Python the language, not Monty Python (depends on context).
- **Popularity vs. freshness:** Should a 5-year-old article with thousands of views rank above a recent article with better content?
- **Domain-specific vocabulary:** Medical search where "MI" means myocardial infarction, not Michigan.

**Approach:** Start with BM25 defaults. Collect search query logs and click-through data. Identify queries with poor results. Iteratively tune: add synonyms, boost certain fields, adjust scoring functions. Modern teams use Learning to Rank (LTR) — machine learning models trained on click data to optimize ranking.

### 3. Scaling Reads

Search clusters often face bursty traffic — a product launch, a news event, Black Friday. Strategies:

- **Replica shards:** Elasticsearch supports replicas. If Shard 1 has 2 replicas, search queries are distributed across all 3 copies.
- **Caching:** Cache frequent queries at the application level. The same "iPhone 16 case" search happening 10,000 times per minute does not need to hit the search cluster every time.
- **Query throttling:** Protect the cluster from overwhelming queries. Rate-limit expensive queries (wildcards, deep aggregations).

### 4. Index Size Management

Search indexes can grow large. A 100GB database might produce a 300GB search index (inverted indexes, stored fields, doc values all add overhead).

- **Index only searchable fields:** Do not index raw HTML, metadata, or fields users never search.
- **Retention policies:** For time-based data (logs, tweets), delete indexes older than N days.
- **Compression:** Modern search engines compress postings lists and stored fields. Elasticsearch's `best_compression` codec reduces storage by 20-30%.

---

## Decision Framework: Do You Need a Search Engine?

Not every application needs Elasticsearch. Here is a decision tree:

```
Do you need a dedicated search engine?

Start here: What kind of search does your app need?
│
├── Exact lookups only (by ID, email, status)
│   └── NO — your database is fine. Use indexes.
│
├── Simple text search on 1-2 fields, <1M records
│   └── MAYBE — try PostgreSQL full-text search first.
│       └── If response time < 200ms, stay with DB.
│       └── If too slow or results are poor, add search engine.
│
├── Full-text search across many fields, >1M records
│   └── YES — add a search engine.
│
├── Need autocomplete / search-as-you-type
│   └── YES — search engines excel here.
│
├── Need faceted search (filters with counts)
│   └── YES — databases cannot do this efficiently.
│
├── Need fuzzy matching / typo tolerance
│   └── YES — database LIKE queries won't cut it.
│
├── Need relevance ranking (best results first)
│   └── YES — databases return results in storage order.
│
└── Need search to scale independently from database
    └── YES — separate search cluster is the right call.
```

### Cost of Adding a Search Engine

| Aspect | Impact |
|--------|--------|
| **Infrastructure** | Additional cluster to run and monitor (3+ nodes for HA) |
| **Data sync** | Must build and maintain CDC pipeline |
| **Consistency** | Search results may be seconds behind the database |
| **Operational** | Index management, mapping changes, cluster upgrades |
| **Expertise** | Team needs search-specific knowledge (analyzers, mappings, scoring) |

**Rule of thumb:** If your users type free-text queries and expect ranked results, you need a search engine. If they click through structured filters and dropdowns, you might not.

---

## Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Inverted index** | Maps terms to documents (opposite of a database index). Enables sub-second full-text search over billions of documents. |
| **Text processing** | Tokenization, stemming, and normalization ensure "Running" matches "run." Pipeline quality directly impacts search quality. |
| **BM25** | Default relevance algorithm. Improves on TF-IDF with term frequency saturation and length normalization. |
| **Search architecture** | Search runs as a separate service. CDC keeps it in sync with the database. Expect 1-5 second lag. |
| **Sharding** | Document-partitioned is the standard. Every query hits all shards (scatter-gather). |
| **Elasticsearch** | Default choice for production search at scale. Powerful but operationally heavy. |
| **Lightweight options** | Meilisearch and Typesense for simpler use cases with built-in typo tolerance. |
| **When to add search** | Free-text queries, autocomplete, faceted filtering, fuzzy matching, or relevance ranking. |

---

## Further Reading

**Related Topics:**

- [Database Indexing](../databases/indexing.md) — B-tree and hash indexes in databases
- [Sharding](../databases/sharding.md) — How databases partition data (similar concepts apply to search)
- [Caching Strategies](../caching/strategies.md) — Caching search results for performance
- [Scalability Patterns](../../scalability/patterns.md) — Scaling read-heavy workloads
