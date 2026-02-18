# Unique ID Generation in Distributed Systems

Every record in a system needs an identifier. In a single database, auto-increment handles this effortlessly: the database locks a counter, increments it, and returns the next value. This guarantee breaks down the moment you have more than one database node. Two nodes cannot share a counter without coordination, and coordination at scale is either slow or complex. Unique ID generation is one of those problems that seems trivial until you try to do it across multiple machines, data centers, and time zones at millions of operations per second.

```
SINGLE DATABASE: Auto-increment works fine

  Client A ──> ┌──────────────┐ ──> ID = 1
  Client B ──> │   Database   │ ──> ID = 2
  Client C ──> │  (one lock)  │ ──> ID = 3
               └──────────────┘
               Sequential, no conflicts


DISTRIBUTED SYSTEM: Auto-increment fails

  Client A ──> ┌─────────┐ ──> ID = 1  ┐
               │  Node 1 │              │ CONFLICT!
  Client B ──> └─────────┘              │
                                        │
  Client C ──> ┌─────────┐ ──> ID = 1  ┘
               │  Node 2 │
  Client D ──> └─────────┘ ──> ID = 2  (also conflicts with Node 1)

  No shared counter, no shared lock. IDs collide.
```

The requirements for distributed ID generation vary by use case, but most systems need some combination of: global uniqueness without coordination, rough time ordering for sorting, compactness for storage efficiency, and high throughput for write-heavy workloads.

---

=== "UUID"

    ## UUID (Universally Unique Identifier)

    UUIDs are 128-bit identifiers standardized by RFC 4122. They are the simplest solution because every node generates IDs independently with no coordination at all. The probability of collision is astronomically small -- for UUIDv4, you would need to generate about 2.71 quintillion IDs before reaching a 50% collision probability.

    ### Structure (128 bits total)

    ```
    UUID Format: xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx

    Example (v4): 550e8400-e29b-41d4-a716-446655440000
                         ^    ^
                         |    N = variant (8, 9, a, or b)
                         M = version (4 for random)

    Bit layout:
    ┌────────────────────────────────────────────────────────┐
    │  32 bits  │ 16 bits │ 16 bits │ 8+8 bits │  48 bits   │
    │ time_low  │ time_mid│ version │ variant  │   node     │
    │           │         │ + time  │ + clock  │            │
    └────────────────────────────────────────────────────────┘
      Total: 128 bits = 16 bytes = 36 characters with hyphens
    ```

    ### UUID Versions That Matter

    **UUIDv1 -- Timestamp + MAC address**
    Uses the current timestamp (100-nanosecond intervals since October 15, 1582) and the generating machine's MAC address. Guarantees uniqueness through time and space, but leaks the machine identity and creation time. Largely deprecated for new designs due to privacy concerns.

    **UUIDv4 -- Random**
    122 bits of randomness (6 bits reserved for version and variant). The most widely used version. No information leakage, but completely unsortable -- inserting into a B-tree index causes random page splits, degrading database write performance over time.

    ```
    B-tree impact of random UUIDs vs sequential IDs:

    Sequential IDs:              Random UUIDs:
    ┌─────────────────┐          ┌─────────────────┐
    │ 1, 2, 3, 4, 5   │         │ a3f, 01b, ff2   │
    │ (always append   │         │ (random inserts  │
    │  to the right)   │         │  everywhere)     │
    └─────────────────┘          └─────────────────┘
    Write amplification: LOW      Write amplification: HIGH
    Page splits: RARE             Page splits: FREQUENT
    Cache hits: HIGH              Cache hits: LOW
    ```

    **UUIDv7 -- Timestamp + Random (newest, RFC 9562)**
    The first 48 bits are a Unix timestamp in milliseconds, followed by 74 bits of randomness. This makes UUIDv7 both globally unique and lexicographically sortable by creation time. It solves the B-tree fragmentation problem of UUIDv4 while keeping the same 128-bit format. This is the recommended UUID version for new systems as of 2024.

    ```
    UUIDv7 structure:
    ┌──────────────────────┬───────┬──────────────────────────┐
    │    48 bits           │ 4 bits│     74 bits              │
    │  Unix timestamp (ms) │version│    Random                │
    └──────────────────────┴───────┴──────────────────────────┘
    │<── sortable ────────>│       │<── unique ──────────────>│
    ```

    **UUID trade-offs:**

    | Aspect           | Assessment                                     |
    |------------------|------------------------------------------------|
    | Uniqueness       | Practically guaranteed (2^122 space for v4)    |
    | Coordination     | None needed -- fully decentralized             |
    | Sortability      | v4: None. v7: Yes, by timestamp                |
    | Size             | 128 bits -- double the size of Snowflake IDs   |
    | Index efficiency | v4: Poor (random). v7: Good (time-ordered)     |
    | Information leak | v1: MAC + time. v4: None. v7: Timestamp only   |

=== "Snowflake ID"

    ## Twitter Snowflake ID

    Twitter created Snowflake in 2010 to generate unique IDs at scale for tweets. The core insight is packing a timestamp, machine identity, and a local sequence counter into a single 64-bit integer. This gives you time-ordered, unique IDs without any coordination between machines.

    ### 64-Bit Structure

    ```
    Snowflake ID (64 bits):
    ┌───┬──────────────────────────────┬──────────┬──────────┬──────────────┐
    │ 0 │         41 bits              │  5 bits  │  5 bits  │   12 bits    │
    │   │       Timestamp              │Datacenter│ Machine  │   Sequence   │
    │   │    (ms since epoch)          │   ID     │   ID     │   Number     │
    └───┴──────────────────────────────┴──────────┴──────────┴──────────────┘
    │   │                              │          │          │              │
    │   │ 2^41 ms = ~69 years          │ 32 DCs   │32 machines│ 4096 IDs/ms │
    │   │ from custom epoch            │ per DC   │ per DC   │ per machine  │
    │   │                              │          │          │              │
    └───┴──────────────────────────────┴──────────┴──────────┴──────────────┘

    Unused bit (always 0, keeps ID positive in signed 64-bit integers)
    ```

    ### How It Works Step by Step

    ```
    1. Read current timestamp (milliseconds since custom epoch)
       Epoch example: Twitter uses Nov 4, 2010
       Current offset: 1,318,000,000,000 ms  (41 bits)

    2. Identify this machine
       Datacenter ID: 3                       (5 bits)
       Machine ID: 12                         (5 bits)

    3. Get next sequence number for this millisecond
       Sequence: 7                            (12 bits)

    4. Combine by bit-shifting:
       ┌─────────────────────────────────────────────────────────────┐
       │ 0 │ 00001001100...01000000 │ 00011 │ 01100 │ 000000000111 │
       └─────────────────────────────────────────────────────────────┘
       Result: 1,541,815,603,606,036,487 (fits in a 64-bit long)

    5. If sequence overflows 4096 in the same millisecond:
       WAIT until the next millisecond, then reset sequence to 0
    ```

    ### Capacity Math

    Each machine can generate **4,096 IDs per millisecond** (12-bit sequence).
    That is **4,096,000 IDs per second per machine**.
    With 1,024 machines (32 DCs x 32 machines): **~4 billion IDs per second** system-wide.

    The 41-bit timestamp gives ~69 years from the custom epoch. Twitter's epoch of November 4, 2010, means Snowflake IDs remain valid until approximately 2079.

    ### Advantages

    - **64 bits** -- fits in a database `BIGINT`, half the storage of UUIDs
    - **Time-sorted** -- IDs created later are always numerically larger
    - **No coordination** -- each machine generates independently
    - **High throughput** -- over 4 million IDs/second per machine
    - **Embeds creation time** -- extract timestamp without a database query

    ### Limitations

    - **Clock dependency** -- if a machine's clock moves backward, IDs could collide or the generator must halt until the clock catches up
    - **Fixed topology** -- 5+5 bit split assumes a specific datacenter/machine layout; changing the topology requires a different bit allocation
    - **Information leakage** -- timestamps reveal when IDs were created and sequence numbers reveal generation rate
    - **Coordination for machine IDs** -- each machine needs a unique datacenter + machine ID, typically assigned via ZooKeeper or configuration

=== "ULID"

    ## ULID (Universally Unique Lexicographically Sortable Identifier)

    ULID was designed as a drop-in replacement for UUIDv4 that adds time-based sortability. It uses Crockford's Base32 encoding, which is case-insensitive and avoids ambiguous characters (I, L, O, U).

    ### Structure

    ```
    ULID: 01ARZ3NDEKTSV4RRFFQ69G5FAV
          |----------|--------------|
           Timestamp     Randomness
           (48 bits)     (80 bits)

    Bit layout (128 bits total):
    ┌──────────────────────────┬──────────────────────────────────────┐
    │        48 bits           │              80 bits                 │
    │  Unix timestamp (ms)     │           Cryptographic random      │
    └──────────────────────────┴──────────────────────────────────────┘
    │<── 10 characters ──────>│<───── 16 characters ───────────────>│

    Encoded as 26 Crockford Base32 characters
    String sort order == chronological order
    ```

    ### ULID vs UUID Comparison

    ```
    UUID:  550e8400-e29b-41d4-a716-446655440000   (36 chars, hex + hyphens)
    ULID:  01ARZ3NDEKTSV4RRFFQ69G5FAV            (26 chars, Base32)

    Both are 128 bits internally, but:
    - ULID string is 10 characters shorter
    - ULID is lexicographically sortable as a string
    - ULID is case-insensitive
    - ULID is compatible with UUID storage (same 128-bit binary)
    ```

    ### Monotonicity Within a Millisecond

    A key ULID design feature: if multiple ULIDs are generated in the same millisecond, the random portion is incremented by 1 rather than regenerated. This preserves sort order even within the same millisecond.

    ```
    Same millisecond, three ULIDs:
    01ARZ3NDEK | SV4RRFFQ69G5FAV  (random = X)
    01ARZ3NDEK | SV4RRFFQ69G5FAW  (random = X + 1)
    01ARZ3NDEK | SV4RRFFQ69G5FAX  (random = X + 2)
    ──────────   ────────────────
    timestamp    monotonic random
    (same)       (incrementing)
    ```

    This means ULIDs are strictly monotonic within a single generator. Across generators, ULIDs from the same millisecond are ordered randomly (by their independent random components), but ULIDs from different milliseconds are always correctly time-ordered.

=== "Database Ranges"

    ## Database Auto-Increment with Range Allocation

    Instead of abandoning auto-increment entirely, you can make it work in a distributed system by pre-allocating ranges of IDs to each node. A central authority hands out non-overlapping blocks, and each node generates sequential IDs within its assigned block.

    ### How It Works

    ```
    Central Authority (e.g., ZooKeeper, or a dedicated DB table)
    ┌────────────────────────────────────────┐
    │  Next available range start: 300,001   │
    │  Block size: 100,000                   │
    └────────────┬─────────────┬─────────────┘
                 │             │
       "Give me  │             │  "Give me
        a block" │             │   a block"
                 ▼             ▼
    ┌─────────────────┐  ┌─────────────────┐
    │     Node A      │  │     Node B      │
    │ Range: 1-100K   │  │ Range: 100K-200K│
    │ Next ID: 87,432 │  │ Next ID: 134,891│
    └─────────────────┘  └─────────────────┘
    Sequential within     Sequential within
    each node             each node
    ```

    ### Trade-offs

    **Advantages:**
    - IDs are simple integers -- small, fast, indexable
    - Roughly ordered (within a node, perfectly ordered)
    - No clock dependency
    - Very high throughput within a block (no network calls)

    **Disadvantages:**
    - IDs are NOT globally ordered across nodes (Node B's 100,001 may be created before Node A's 50,000)
    - Central authority is a single point of failure (mitigated by making it highly available)
    - Gaps in ID space when a node crashes with unused IDs in its range
    - Range exhaustion requires a network round-trip to get a new block

=== "Ticket Servers"

    ## Database Ticket Servers (Flickr Approach)

    Flickr solved ID generation in 2010 using dedicated MySQL instances as "ticket servers." Each server uses MySQL's `auto_increment_increment` and `auto_increment_offset` to generate non-overlapping sequences.

    ### Architecture

    ```
    Two ticket servers generating interleaved IDs:

    Ticket Server 1 (odd IDs)          Ticket Server 2 (even IDs)
    auto_increment_increment = 2        auto_increment_increment = 2
    auto_increment_offset = 1           auto_increment_offset = 2
    ┌─────────────────────┐             ┌─────────────────────┐
    │ REPLACE INTO Tickets│             │ REPLACE INTO Tickets│
    │ (stub) VALUES ('a') │             │ (stub) VALUES ('a') │
    │                     │             │                     │
    │ Returns: 1,3,5,7,9  │             │ Returns: 2,4,6,8,10 │
    └─────────────────────┘             └─────────────────────┘
              │                                   │
              └───────────┐           ┌───────────┘
                          ▼           ▼
                   ┌─────────────────────┐
                   │   Load Balancer /   │
                   │   Round-Robin       │
                   │                     │
                   │  App gets IDs from  │
                   │  either server      │
                   └─────────────────────┘
    ```

    ### How the REPLACE Trick Works

    Flickr uses a single-row table with `REPLACE INTO`, which deletes the existing row and inserts a new one, triggering auto-increment each time. The result is a lightweight, high-performance ID generator that piggybacks on MySQL's battle-tested auto-increment machinery.

    ```
    CREATE TABLE Tickets64 (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      stub CHAR(1) NOT NULL DEFAULT '',
      PRIMARY KEY (id),
      UNIQUE KEY stub (stub)
    );

    -- Each call returns the next ID:
    REPLACE INTO Tickets64 (stub) VALUES ('a');
    SELECT LAST_INSERT_ID();
    ```

    ### Trade-offs

    - **Simple and proven** -- relies on MySQL, which is extremely well understood
    - **Highly available** -- two servers with round-robin gives redundancy
    - **Roughly sortable** -- IDs increase over time (though odd/even interleaving)
    - **Single point of failure risk** -- if both ticket servers go down, no IDs can be generated
    - **Throughput ceiling** -- limited by MySQL's write speed (typically 10K-50K IDs/second per server)
    - **Network dependency** -- every ID requires a database round-trip

=== "Sonyflake"

    ## Sonyflake

    Sonyflake is Sony's alternative to Snowflake, optimized for longer lifespans and more machines at the cost of lower per-machine throughput. It reprioritizes the bit allocation to favor a longer timestamp and more machine IDs.

    ### 64-Bit Structure

    ```
    Sonyflake ID (64 bits):
    ┌──────────────────────────────────────┬──────────┬──────────────────┐
    │              39 bits                 │  8 bits  │    16 bits       │
    │           Timestamp                  │ Sequence │   Machine ID     │
    │     (10ms units since epoch)         │  Number  │                  │
    └──────────────────────────────────────┴──────────┴──────────────────┘
    │                                      │          │                  │
    │ 2^39 * 10ms = ~174 years             │ 256/10ms │  65,536 machines │
    │                                      │          │                  │
    └──────────────────────────────────────┴──────────┴──────────────────┘
    ```

    ### Snowflake vs Sonyflake Bit Allocation

    ```
    Snowflake:  [1] [  41 bits timestamp  ] [5+5 datacenter+machine] [12 seq]
    Sonyflake:  [   39 bits timestamp     ] [  8 bits sequence     ] [16 machine]

    ┌──────────────────┬───────────────┬──────────────┐
    │                  │   Snowflake   │  Sonyflake   │
    ├──────────────────┼───────────────┼──────────────┤
    │ Timestamp bits   │   41          │   39         │
    │ Time unit        │   1 ms        │   10 ms      │
    │ Lifespan         │   ~69 years   │   ~174 years │
    │ Machine ID bits  │   10          │   16         │
    │ Max machines     │   1,024       │   65,536     │
    │ Sequence bits    │   12          │   8          │
    │ IDs per unit     │   4,096/ms    │   256/10ms   │
    │ IDs/sec/machine  │   4,096,000   │   25,600     │
    └──────────────────┴───────────────┴──────────────┘
    ```

    Sonyflake is ideal when you have many machines (IoT, edge computing) and do not need extreme per-machine throughput, but want the ID scheme to last well over a century.

---

## Comparison: All Approaches

| Approach | Size | Sortable | Coordination | IDs/sec/node | Info Leaked | Best For |
|----------|------|----------|-------------|-------------|-------------|----------|
| UUIDv4 | 128 bits | No | None | Unlimited | Nothing | Simple, low-volume systems |
| UUIDv7 | 128 bits | Yes | None | Unlimited | Timestamp | New systems needing UUID compat |
| Snowflake | 64 bits | Yes | Machine ID assignment | ~4M | Timestamp + rate | High-throughput, time-ordered |
| ULID | 128 bits | Yes | None | Unlimited | Timestamp | UUID replacement with sorting |
| DB Ranges | 64 bits | Partial | Range allocation | Very high | Approximate volume | Existing SQL infrastructure |
| Ticket Server | 64 bits | Rough | Per-ID (network call) | 10K-50K | Volume + ordering | Simple HA setups (Flickr) |
| Sonyflake | 64 bits | Yes | Machine ID assignment | ~25K | Timestamp + rate | Many machines, long lifespan |
| MongoDB ObjectID | 96 bits | Yes | None | ~16M | Timestamp + machine | MongoDB-native applications |

---

## Design Considerations

=== "Clock Problems"

    ## Clock Synchronization Issues

    Any ID scheme that embeds a timestamp is vulnerable to clock problems. In a distributed system, machine clocks are never perfectly synchronized, and they can drift or jump.

    ### Types of Clock Failures

    ```
    Normal operation:
    Node A clock:  ──────────────────────────>  (t=1000, t=1001, t=1002...)
    Node B clock:  ──────────────────────────>  (t=1000, t=1001, t=1002...)
                   Clocks drift ~10-100ms apart over hours

    NTP correction (clock jumps backward):
    Node A clock:  ────────────────┐
                                   │ NTP says "you're 200ms ahead"
                   ────────────────┘──────────>
                   t=1000...t=1200, JUMP to t=1000, t=1001...

    Leap second (clock jumps or smears):
    Node A clock:  ──────23:59:59──|──00:00:00──>  (normal)
    Node B clock:  ──────23:59:59──23:59:60──00:00:00──>  (leap second)
    ```

    ### Impact on ID Generation

    **Snowflake/Sonyflake:** If the clock moves backward, the generator detects this and either refuses to generate IDs until the clock catches up (causing downtime) or uses a fallback strategy. Twitter's Snowflake logs an error and waits.

    **UUIDv7/ULID:** A backward clock jump means newly generated IDs have timestamps earlier than previously generated IDs, breaking sort order. ULIDv7 implementations typically keep track of the last timestamp and increment the random portion if the clock moves backward.

    ### Mitigations

    - **Google TrueTime:** Uses GPS and atomic clocks to bound clock uncertainty to < 7ms. Spanner waits out the uncertainty window before committing. Expensive, but eliminates clock ambiguity.
    - **Hybrid Logical Clocks (HLC):** Combine physical timestamps with logical counters. If the physical clock moves backward, the logical counter increments instead, preserving ordering.
    - **Monotonic clock sources:** Use the OS monotonic clock (which never goes backward) for measuring elapsed time, even though it cannot be compared across machines.

=== "Monotonicity"

    ## Monotonicity Guarantees

    Monotonicity means IDs always increase. This matters for database indexing, event ordering, and systems that use IDs as a proxy for "happened before."

    ```
    Strict monotonicity (single generator):
    ID:  1  2  3  4  5  6  7  8  9  10
         ────────────────────────────────>  Always increasing

    Rough monotonicity (distributed, timestamp-based):
    Node A: 100, 103, 107, 110, 115
    Node B: 101, 104, 106, 112, 114
    Node C: 102, 105, 108, 111, 113
    Merged:  Mostly increasing, occasional out-of-order within ~ms window

    No monotonicity (UUIDv4):
    f47ac10b, 7c9e6679, 1b4e28ba, 550e8400, 9b2d5e7a
    Completely random, no ordering relationship
    ```

    **Levels of ordering guarantee:**

    | Level | Definition | Achieved By |
    |-------|-----------|-------------|
    | Strict monotonic | Every ID > all previous IDs | Single generator, sequence lock |
    | Per-node monotonic | Each node's IDs are strictly increasing | Snowflake, Sonyflake |
    | Roughly monotonic | IDs increase over time with small window of disorder | UUIDv7, ULID (across nodes) |
    | Unordered | No relationship between ID values | UUIDv4 |

    For most use cases, rough monotonicity is sufficient. Strict monotonicity requires a single point of serialization, which limits throughput.

=== "Information Leakage"

    ## Information Leakage

    IDs can reveal more about your system than you intend. A competitor who understands your ID scheme can extract operational intelligence.

    ### What Each Scheme Reveals

    ```
    Given two Snowflake-style IDs from a public API:

    ID 1: 1541815603606036487  (Monday 10:00 AM)
    ID 2: 1541815607806124103  (Monday 10:01 AM)

    An observer can extract:
    ┌────────────────────────────────────────────┐
    │ Timestamp:  Exact creation time (to ms)    │
    │ Machine:    Which server generated it       │
    │ Sequence:   How many IDs that ms            │
    │                                             │
    │ Derived:                                    │
    │ - Your growth rate (IDs between two times) │
    │ - Peak hours (when sequences are highest)  │
    │ - Infrastructure (how many machines)        │
    │ - Datacenter location (from DC ID)         │
    └────────────────────────────────────────────┘
    ```

    **Real example -- the "how many orders" problem:**
    If an e-commerce site uses sequential IDs for orders, a competitor can place an order on Monday (order #50,000) and another on Friday (order #50,500) and learn: "They processed 500 orders this week."

    ### Mitigations

    | Strategy | How It Works | Trade-off |
    |----------|-------------|-----------|
    | Use UUIDv4 for external IDs | Random, reveals nothing | Larger storage, not sortable |
    | Encrypt the ID | Apply format-preserving encryption to the integer ID | CPU overhead, key management |
    | Hash-based obfuscation | External ID = hash(internal ID + secret) | Lookup table needed |
    | Separate internal/external IDs | Internal: Snowflake for efficiency. External: random token | Two ID columns per row |

    Many systems use a dual-ID approach: Snowflake internally for performance and ordering, and a random external token (like Stripe's `ch_1A2B3C4D5E`) for public APIs.

=== "Storage Efficiency"

    ## Storage Efficiency: 64-bit vs 128-bit

    The difference between 64-bit and 128-bit IDs compounds at scale. Every row has an ID, every index entry includes the ID, every foreign key references the ID, and every join operation compares IDs.

    ```
    Storage impact at different scales:

    Rows         | 64-bit IDs | 128-bit IDs | Difference
    ─────────────┼────────────┼─────────────┼──────────
    1 million    | 8 MB       | 16 MB       | 8 MB
    100 million  | 800 MB     | 1.6 GB      | 800 MB
    10 billion   | 80 GB      | 160 GB      | 80 GB

    But with 3 indexes + 2 foreign keys (5x the ID stored):
    10 billion   | 400 GB     | 800 GB      | 400 GB

    Plus replicas (3x): 1.2 TB vs 2.4 TB difference
    ```

    For systems at Twitter or Discord scale (trillions of rows across services), the choice between 64-bit and 128-bit IDs translates to petabytes of storage difference. This is why high-throughput systems almost universally prefer 64-bit Snowflake-style IDs over 128-bit UUIDs.

    However, for systems with fewer than a billion rows, the storage overhead of 128-bit IDs is negligible compared to the simplicity of UUID generation.

---

## Real-World Examples

=== "Twitter"

    ## Twitter: Snowflake

    Twitter developed Snowflake in 2010 when their growth made MySQL auto-increment untenable. At the time, they were migrating from a monolithic MySQL setup to a distributed architecture using Cassandra and Gizzard (their sharding framework).

    ```
    Twitter's ID generation at scale:

    ┌──────────────────────────────────────────────────┐
    │              Twitter Infrastructure               │
    │                                                   │
    │  DC: East Coast         DC: West Coast            │
    │  ┌────────────┐         ┌────────────┐            │
    │  │ Snowflake  │         │ Snowflake  │            │
    │  │ Cluster    │         │ Cluster    │            │
    │  │            │         │            │            │
    │  │ 10+ nodes  │         │ 10+ nodes  │            │
    │  │ ~10K IDs/s │         │ ~10K IDs/s │            │
    │  │ per node   │         │ per node   │            │
    │  └────────────┘         └────────────┘            │
    │                                                   │
    │  Custom epoch: November 4, 2010 (Twitter's birth)│
    │  Machine IDs: assigned via ZooKeeper              │
    │  Throughput: >10,000 IDs/second per node          │
    │  Deployed as: Thrift service (network accessible) │
    └──────────────────────────────────────────────────┘
    ```

    **Key design decisions:**
    - Ran as a standalone Thrift service, not embedded in applications
    - ZooKeeper assigned machine IDs to avoid conflicts
    - Custom epoch maximized the 41-bit timestamp lifespan
    - Open-sourced the design (though not actively maintained)

=== "Instagram"

    ## Instagram: Snowflake-Inspired with PostgreSQL

    Instagram needed globally unique, time-sortable IDs but wanted to stay within PostgreSQL rather than deploy a separate service. They embedded Snowflake-style ID generation directly into PostgreSQL using PL/pgSQL functions and schemas for sharding.

    ```
    Instagram's approach:

    ┌──────────────────────────────────────────┐
    │         PostgreSQL Cluster                │
    │                                           │
    │  Shard 1          Shard 2                 │
    │  ┌─────────┐      ┌─────────┐            │
    │  │ Schema 1│      │ Schema 2│            │
    │  │         │      │         │            │
    │  │ next_id │      │ next_id │            │
    │  │ function│      │ function│            │
    │  └─────────┘      └─────────┘            │
    │                                           │
    │  ID structure (64 bits):                  │
    │  [41 bits: ms since custom epoch]         │
    │  [13 bits: shard ID (8192 shards)]        │
    │  [10 bits: sequence (1024/ms/shard)]      │
    └──────────────────────────────────────────┘

    Custom epoch: January 1, 2011
    Shard IDs:   Derived from PostgreSQL schema
    No external service:  ID generation lives in the DB
    ```

    **Why this works for Instagram:**
    - No separate ID generation service to maintain and monitor
    - Shard ID is embedded in the ID itself, so you can determine which shard holds a row just from its ID
    - PostgreSQL's sequence guarantees monotonicity within a shard
    - At ~1,000 IDs per millisecond per shard, throughput was sufficient for their photo upload rate

=== "Discord"

    ## Discord: Snowflake Variant

    Discord adopted a Snowflake variant for message IDs. Since message ordering is critical for a chat application, time-sortable IDs let Discord sort messages by ID rather than maintaining a separate timestamp index.

    ```
    Discord's requirements:
    ┌────────────────────────────────────────┐
    │ - Millions of messages per second       │
    │ - Messages must sort chronologically    │
    │ - IDs must be unique across all guilds  │
    │ - IDs should fit in 64 bits (JS safe*)  │
    └────────────────────────────────────────┘

    * JavaScript numbers lose precision above 2^53.
      Discord IDs are sent as strings in JSON to avoid this.

    Discord Snowflake structure:
    ┌──────────────────────────────┬───────────┬──────────┬──────────────┐
    │         42 bits              │  5 bits   │  5 bits  │   12 bits    │
    │   Timestamp (ms)             │ Worker ID │Process ID│  Sequence    │
    │   since Discord epoch        │           │          │              │
    └──────────────────────────────┴───────────┴──────────┴──────────────┘

    Discord epoch: January 1, 2015 (first day of Discord)
    ```

    **Key differences from Twitter's Snowflake:**
    - Uses Worker ID + Process ID instead of Datacenter ID + Machine ID
    - Custom epoch of 2015 gives IDs a longer effective lifespan
    - IDs are transmitted as strings in the API to avoid JavaScript precision loss
    - Message sorting is done entirely by ID comparison -- no need for a timestamp column

=== "MongoDB"

    ## MongoDB: ObjectID

    MongoDB's ObjectID is a 96-bit (12-byte) identifier that includes enough information to be generated on any client or server without coordination.

    ```
    MongoDB ObjectID (96 bits / 12 bytes):
    ┌──────────────────┬────────────┬──────────┬──────────────┐
    │     32 bits      │   40 bits  │ 24 bits  │   24 bits    │
    │    Timestamp     │  Random    │ Random   │  Incrementing│
    │   (seconds)      │  (per-process value)  │   Counter    │
    └──────────────────┴────────────┴──────────┴──────────────┘
    │                  │                       │              │
    │ Unix timestamp   │ 5 random bytes        │ Starts at    │
    │ (second          │ (unique per machine   │ random value,│
    │  precision)      │  and process)         │ increments   │
    └──────────────────┴───────────────────────┴──────────────┘

    Example: 507f1f77bcf86cd799439011
             ├──────┤├──────────┤├────┤
             time     random      counter
    ```

    **ObjectID trade-offs:**
    - 96 bits: smaller than UUID (128) but larger than Snowflake (64)
    - Second-precision timestamp: coarser than millisecond schemes
    - Client-generated: the driver creates ObjectIDs, not the server
    - Roughly sortable: sort by ObjectID approximates sort by creation time
    - No configuration needed: random bytes replace machine/datacenter IDs

---

## Decision Framework

Use this flow to choose the right ID generation strategy for your system:

```
START: What are your constraints?
│
├── Need to fit in 64 bits (database BIGINT)?
│   │
│   ├── YES: How many machines generate IDs?
│   │   │
│   │   ├── < 1,024 machines
│   │   │   │
│   │   │   ├── Need > 25K IDs/sec/machine? ──> Twitter Snowflake
│   │   │   │
│   │   │   └── 25K IDs/sec is enough?  ──────> Sonyflake (longer lifespan)
│   │   │
│   │   └── > 1,024 machines ─────────────────> Sonyflake (65K machine IDs)
│   │
│   └── NO (128 bits OK): Do you need UUID compatibility?
│       │
│       ├── YES: Need sortability?
│       │   │
│       │   ├── YES ──────────────────────────> UUIDv7
│       │   │
│       │   └── NO ───────────────────────────> UUIDv4
│       │
│       └── NO: Need shorter string representation?
│           │
│           ├── YES ──────────────────────────> ULID (26 chars vs 36)
│           │
│           └── NO ───────────────────────────> UUIDv7
│
├── Already using PostgreSQL and want no external dependencies?
│   │
│   └── ───────────────────────────────────────> Instagram approach
│       (PL/pgSQL function with embedded shard ID)
│
├── Need absolute simplicity with high availability?
│   │
│   └── ───────────────────────────────────────> Ticket servers (Flickr)
│       (Two MySQL instances with interleaved auto-increment)
│
└── Using MongoDB?
    │
    └── ───────────────────────────────────────> ObjectID (built-in)
```

### Quick Reference by Use Case

| Use Case | Recommended | Why |
|----------|------------|-----|
| General-purpose web app | UUIDv7 | No setup, sortable, universal support |
| High-throughput (>1M IDs/sec) | Snowflake | 64-bit, 4M IDs/sec/node |
| Microservices, no coordination | UUIDv7 or ULID | Zero configuration |
| Chat/messaging (need time sort) | Snowflake variant | Efficient sort by ID |
| IoT with many devices | Sonyflake | 65K machine IDs, 174-year lifespan |
| Event sourcing / audit logs | Snowflake or UUIDv7 | Timestamp embedded, sortable |
| Public API identifiers | UUIDv4 + prefix | Reveals nothing, looks like `usr_550e8400...` |
| Legacy SQL migration | Ticket servers | Minimal change to existing schema |
| MongoDB-native | ObjectID | Built-in, no setup |

---

## Key Takeaways

1. **Auto-increment fails in distributed systems** because there is no shared counter. Every approach to distributed ID generation is a different set of trade-offs around size, sortability, coordination, and information leakage.

2. **Snowflake (64-bit) is the gold standard** for high-throughput systems that need time-sorted IDs. Twitter, Discord, and Instagram all use variants. The 64-bit size saves significant storage at scale.

3. **UUIDv7 is the new default** for systems that do not have extreme throughput requirements. It gives you sortability with zero coordination and universal library support.

4. **Clock synchronization is the Achilles' heel** of timestamp-based schemes. Systems must handle backward clock jumps, NTP corrections, and leap seconds gracefully.

5. **No single scheme is universally best.** A chat application, an e-commerce platform, and an IoT network have fundamentally different requirements. Use the decision framework to match approach to constraints.

---

**Related Topics:**

- [Consistent Hashing](consistent-hashing.md) -- how distributed data placement works
- [Consensus Algorithms](consensus.md) -- coordination mechanisms like ZooKeeper for machine ID assignment
- [Database Sharding](../data/databases/sharding.md) -- partitioning strategies that depend on ID design
- [CAP Theorem](../fundamentals/cap-theorem.md) -- the fundamental trade-off underlying distributed ID generation
