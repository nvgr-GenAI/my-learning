# Data Consistency Models

**Master consistency guarantees in distributed systems** | 🎯 Core Models | ⚖️ Trade-offs | 💼 Interview Ready

## Quick Reference

**Data Consistency** - How and when replicas see the same data across distributed systems:

| Model | Guarantee | Latency | Use Cases | Example Systems |
|-------|-----------|---------|-----------|-----------------|
| **Strong/Linearizable** | All nodes see same data instantly | High (100-500ms) | Banking, inventory, bookings | Spanner, etcd, ZooKeeper |
| **Eventual** | Converges over time (seconds-minutes) | Low (5-50ms) | Social media, catalogs, CDN | DynamoDB, Cassandra, S3 |
| **Causal** | Causally related ops ordered | Medium (50-100ms) | Chat, collaborative editing | COPS, MongoDB sessions |
| **Read-Your-Writes** | User sees own writes immediately | Low (10-50ms) | User profiles, settings | Most web apps |
| **Monotonic Reads** | Data never goes backwards | Low (10-50ms) | Comment threads, feeds | Sticky sessions |
| **Weak** | No guarantees (best effort) | Lowest (<5ms) | Live streaming, leaderboards | Memcached, analytics |

**Key Insight:** Consistency is a spectrum, not binary. Choose the weakest model that satisfies your requirements for better performance.

---

=== "🎯 Understanding Consistency"

    ## What is Data Consistency?

    **Data consistency** determines what data replicas can see and when they see it in distributed systems.

    ### The Library Analogy

    Imagine a library system with 3 branches:

    === "Strong Consistency"

        ```
        📚 Strong Consistency (All branches synchronized)
        ──────────────────────────────────────────────────

        09:00 AM - Customer checks out book at Branch A
        09:01 AM - System locks all branches
        09:02 AM - All branches updated simultaneously
        09:03 AM - Update confirmed, customer can leave
        09:04 AM - Any branch query shows "Checked Out"

        ✅ Perfect accuracy across all locations
        ❌ Slower operations (wait for all branches)
        ❌ System unavailable if any branch is down

        Perfect for: Financial records, legal documents
        ```

    === "Eventual Consistency"

        ```
        📚 Eventual Consistency (Branches sync later)
        ──────────────────────────────────────────────────

        09:00 AM - Customer checks out book at Branch A
        09:01 AM - Branch A immediately records "Checked Out"
        09:02 AM - Customer leaves (fast!)
        09:03 AM - Branch B still shows "Available" (stale!)
        09:05 AM - Branch C still shows "Available" (stale!)
        09:10 AM - Sync completes, all branches updated
        09:11 AM - All branches now show "Checked Out"

        ✅ Fast operations (don't wait for other branches)
        ✅ System works even if branches are disconnected
        ❌ Temporary inconsistency (5-10 minute window)

        Perfect for: Popular book lists, recommendations
        ```

    ---

    ## The Consistency Spectrum

    ```mermaid
    graph LR
        subgraph "Stronger Guarantees → Slower, More Expensive"
        A[Strong/Linearizable<br/>💰 Banking<br/>📦 Inventory]
        B[Sequential<br/>📋 Ordered logs]
        C[Causal<br/>💬 Chat<br/>📝 Docs]
        end

        subgraph "Weaker Guarantees → Faster, More Scalable"
        D[Eventual<br/>📱 Social Media<br/>🌐 CDN]
        E[Weak<br/>📺 Live Video<br/>📊 Analytics]
        end

        A -->|Relax ordering| B
        B -->|Only causal order| C
        C -->|Allow staleness| D
        D -->|No guarantees| E

        style A fill:#ff6b6b
        style E fill:#51cf66
    ```

    ### Why Consistency Matters

    | Scenario | Wrong Choice | Impact | Right Choice |
    |----------|-------------|--------|--------------|
    | **Banking Transfer** | Eventual consistency | Money duplicated or lost | Strong consistency ✓ |
    | **Inventory System** | Weak consistency | Oversell products, angry customers | Strong consistency ✓ |
    | **Social Media Feed** | Strong consistency | Slow loading, poor UX | Eventual consistency ✓ |
    | **Live Leaderboard** | Strong consistency | High latency, lag | Weak consistency ✓ |
    | **Chat Messages** | Eventual consistency | Reply appears before message | Causal consistency ✓ |

    ### The Cost of Consistency

    ```
    Strong Consistency:
    ───────────────────
    Write: 200ms (wait for all replicas)
    Read:  20ms (any replica)
    Cost:  High (coordination overhead)
    Scale: Limited (synchronous bottleneck)

    Eventual Consistency:
    ────────────────────
    Write: 10ms (local only)
    Read:  5ms (nearest replica)
    Cost:  Low (async replication)
    Scale: Excellent (independent regions)

    Trade-off: 20x faster, but temporary staleness
    ```

=== "📊 Consistency Models"

    ## The Six Key Models

    === "1. Strong Consistency"

        ### Definition

        **Every read returns the most recent write. All replicas see the same data simultaneously.**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | System behaves like single copy |
        | **Visibility** | Write is atomic (all or none) |
        | **Ordering** | Total global order of operations |
        | **Reads** | Never stale |

        ---

        ### How It Works

        ```
        Timeline: Bank Transfer $100
        ─────────────────────────────────────────────

        T0: Client initiates transfer
        T1: Lock both accounts on ALL replicas
        T2: Validate balances (atomic read)
        T3: Update Account A: -$100
        T4: Update Account B: +$100
        T5: Commit to ALL replicas
        T6: Release locks
        T7: Acknowledge success

        KEY: Steps 1-6 block other operations
             No partial states visible
             High latency but perfect correctness
        ```

        ---

        ### Real-World Examples

        **Banking System:**
        ```
        Requirement: Account balance must be accurate
        Why Strong: $1000 transfer can't result in $1100

        Without Strong Consistency:
        ─────────────────────────────
        Replica A: Balance = $1000
        Replica B: Balance = $1000 (not yet synced)
        User withdraws $800 from A ✓
        User withdraws $800 from B ✓
        Result: Both succeed! Overdraft by $600 ❌

        With Strong Consistency:
        ────────────────────────
        Replica A: Balance = $1000
        Replica B: Balance = $1000 (synchronized)
        User withdraws $800 from A ✓
        User withdraws $800 from B ❌ (sees updated balance $200)
        Result: Second withdrawal fails ✓
        ```

        **Inventory Management:**
        ```
        Requirement: Can't oversell products
        Why Strong: Only 1 item left in stock

        Scenario: Black Friday flash sale, 1 item remaining
        Without Strong: 100 customers click "Buy" simultaneously
                       All see "In Stock", all orders succeed ❌
        With Strong: First customer gets it, other 99 see "Sold Out" ✓
        ```

        ---

        ### Trade-offs

        **Pros:**
        - ✅ Simple to reason about (like single database)
        - ✅ No stale data ever
        - ✅ No conflict resolution needed
        - ✅ Meets compliance requirements

        **Cons:**
        - ❌ High latency (200-500ms for global sync)
        - ❌ Lower availability (fails if replicas down)
        - ❌ Scalability limits (synchronous bottleneck)
        - ❌ Higher infrastructure cost

        ---

        ### When to Use

        | Scenario | Why Strong Consistency | Consequence of Failure |
        |----------|----------------------|------------------------|
        | **Financial transactions** | Money accuracy critical | Legal liability, lost funds |
        | **Inventory/stock management** | Prevent overselling | Customer refunds, reputation damage |
        | **Seat/hotel bookings** | No double-booking | Service disruption, refunds |
        | **Distributed locks** | Mutual exclusion required | Data corruption |
        | **Legal/audit records** | Immutable history | Compliance violations |

        ---

        ### Example Systems

        | System | Consistency Mechanism | Global? |
        |--------|---------------------|---------|
        | **Google Spanner** | TrueTime + 2PC | Yes (worldwide) |
        | **CockroachDB** | Raft consensus | Yes |
        | **etcd** | Raft consensus | Single cluster |
        | **ZooKeeper** | ZAB protocol | Single cluster |
        | **Traditional RDBMS** | ACID transactions | Single node |

    === "2. Eventual Consistency"

        ### Definition

        **All nodes will eventually see the same data, given enough time without new writes.**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | Convergence (not immediate) |
        | **Visibility** | Write visible locally, then propagates |
        | **Time Window** | Seconds to minutes (configurable) |
        | **Conflicts** | Possible, need resolution strategy |

        ---

        ### How It Works

        ```
        Timeline: Social Media Post
        ─────────────────────────────────────────────

        T0: User creates post in US datacenter
        T1: US datacenter writes locally (5ms) ✓
        T2: US users see post immediately ✓
        T3: Async replication queued
        T4: EU users read → don't see post yet (stale)
        T5: Asia users read → don't see post yet (stale)
        T10: Replication reaches EU (sync time ~10s)
        T15: Replication reaches Asia (sync time ~15s)
        T16: All users now see the post ✓

        KEY: Fast writes, temporary staleness
             Eventually consistent after sync
        ```

        ---

        ### Conflict Resolution Strategies

        **1. Last-Write-Wins (LWW)**
        ```
        Conflict: User updates profile from 2 locations
        ─────────────────────────────────────────────
        US:   Name = "John Smith"    (timestamp: 1000)
        EU:   Name = "J. Smith"      (timestamp: 1002)

        Resolution: EU wins (later timestamp)
        Result: Name = "J. Smith" everywhere

        Pros: Simple, automatic
        Cons: Loses US update (data loss)
        Use: Profile updates, settings
        ```

        **2. Application-Defined Merge**
        ```
        Conflict: User adds items to cart from 2 devices
        ─────────────────────────────────────────────
        Mobile: Cart = [Item A, Item B]
        Web:    Cart = [Item B, Item C]

        Resolution: Union merge
        Result: Cart = [Item A, Item B, Item C]

        Pros: No data loss, semantic merge
        Cons: Requires custom logic
        Use: Shopping carts, collaborative editing
        ```

        **3. Vector Clocks (Detect Causality)**
        ```
        Tracks which writes happened-before others
        ─────────────────────────────────────────────
        If A happened-before B: Keep B, discard A
        If A and B are concurrent: Conflict! Let app decide

        Pros: Detects true conflicts
        Cons: Complex, storage overhead
        Use: Distributed databases (Riak, Cassandra)
        ```

        ---

        ### Real-World Examples

        **Social Media Feed:**
        ```
        Requirement: Fast loading, global reach
        Why Eventual: Users tolerate 10-second post delay

        User Experience:
        ────────────────
        Post creation: Instant (5ms)
        Own feed: See immediately
        Friends' feeds: Appear in 5-30 seconds
        Global feed: Appear in 30-60 seconds

        Acceptable because:
        - Speed matters more than instant consistency
        - Posts aren't time-critical
        - Users don't notice 10-second delay
        ```

        **DNS System:**
        ```
        Requirement: Global availability
        Why Eventual: DNS updates take 24-48 hours

        Update DNS record:
        ────────────────
        T0: Update at primary nameserver
        T1: Propagates to secondary nameservers (hours)
        T2: Eventually reaches all DNS servers globally

        Acceptable because:
        - DNS changes are infrequent
        - Old IP continues working during transition
        - Users don't notice gradual rollout
        ```

        ---

        ### Trade-offs

        **Pros:**
        - ✅ Low latency (5-50ms for local writes)
        - ✅ High availability (no coordination)
        - ✅ Better partition tolerance
        - ✅ Excellent scalability (independent regions)

        **Cons:**
        - ❌ Temporary inconsistency (seconds to minutes)
        - ❌ Conflict resolution complexity
        - ❌ Application must handle stale reads
        - ❌ No guarantees on consistency window

        ---

        ### When to Use

        | Scenario | Why Eventual | Staleness Window |
        |----------|--------------|------------------|
        | **Social media posts** | Speed > perfect consistency | 5-30 seconds OK |
        | **Product catalogs** | Price/description updates | 1-5 minutes OK |
        | **User profiles** | Display name, avatar | 10-60 seconds OK |
        | **CDN content** | Images, videos, static files | 5-15 minutes OK |
        | **DNS records** | Domain name resolution | 24-48 hours OK |

        ---

        ### Example Systems

        | System | Consistency Window | Conflict Resolution |
        |--------|-------------------|---------------------|
        | **Amazon DynamoDB** | Configurable (100ms-1s) | Last-write-wins or custom |
        | **Cassandra** | Tunable (seconds) | Last-write-wins, timestamps |
        | **Amazon S3** | Seconds to minutes | Last-write-wins |
        | **CouchDB** | Seconds | Multi-version, app resolves |
        | **Riak** | Configurable | Vector clocks, siblings |

    === "3. Causal Consistency"

        ### Definition

        **Operations that are causally related are seen in order. Concurrent operations can be reordered.**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | If A causes B, everyone sees A before B |
        | **Concurrent ops** | Can be seen in any order |
        | **Ordering** | Partial order (not total) |
        | **Mechanism** | Vector clocks, dependency tracking |

        ---

        ### How It Works

        ```
        Chat Thread Example
        ─────────────────────────────────────────────

        Message 1: "What's for dinner?" (Alice)
        Message 2: "Pizza!" (Bob, reply to #1)
        Message 3: "I love pizza!" (Carol, reply to #2)
        Message 4: "Who won the game?" (Dave, unrelated)

        Causal Dependencies:
        ────────────────────
        #1 → #2 → #3  (causal chain)
        #4 (concurrent with all)

        Valid Orderings:
        ✓ #1, #2, #3, #4
        ✓ #1, #4, #2, #3
        ✓ #4, #1, #2, #3
        ✗ #2, #1, #3, #4  (reply before question!)
        ✗ #1, #3, #2, #4  (reply before answer!)

        KEY: Causal order preserved, unrelated messages flexible
        ```

        ---

        ### Real-World Examples

        **Chat/Messaging:**
        ```
        Requirement: Replies must appear after original
        Why Causal: Preserve conversation flow

        Without Causal:
        ──────────────
        User A: "What's the meeting time?"
        User B: "3pm"
        User C sees: "3pm" then "What's the meeting time?" ❌
        Confusing!

        With Causal:
        ───────────
        User C always sees question before answer ✓
        But unrelated messages can be reordered ✓
        ```

        **Collaborative Editing:**
        ```
        Requirement: Dependent edits must be ordered
        Why Causal: Maintain document coherence

        Document: "The cat sat on the mat"

        Edit 1: Change "cat" to "dog"
        Edit 2: Add "brown" before "dog" (depends on Edit 1)

        Causal ensures: Edit 2 never applied before Edit 1
        Result: "The brown dog sat on the mat" ✓
        ```

        ---

        ### Trade-offs

        **Pros:**
        - ✅ Preserves meaningful relationships
        - ✅ Faster than strong consistency
        - ✅ More intuitive than eventual
        - ✅ Natural for interactive systems

        **Cons:**
        - ❌ Complex implementation
        - ❌ Overhead of tracking dependencies
        - ❌ Delayed message delivery
        - ❌ Storage overhead (vector clocks)

        ---

        ### When to Use

        | Use Case | Why Causal | Non-Causal Impact |
        |----------|-----------|-------------------|
        | **Chat systems** | Reply ordering critical | Confusing conversations |
        | **Social media comments** | Thread structure matters | Broken threads |
        | **Collaborative editing** | Edit dependencies | Document corruption |
        | **Git/version control** | Commit dependencies | Broken history |

        ---

        ### Example Systems

        - **COPS** (Clusters of Order-Preserving Servers)
        - **Eiger** (Causal consistency for geo-replication)
        - **MongoDB** (causal consistency within sessions)
        - **Riak** (causal context support)

    === "4. Read-Your-Writes"

        ### Definition

        **A user always sees their own updates immediately (but may not see others' updates).**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | Per-user consistency |
        | **Scope** | Only for the writing user |
        | **Others' Writes** | May be stale |
        | **Implementation** | Session tracking, sticky sessions |

        ---

        ### How It Works

        ```
        User Profile Update
        ─────────────────────────────────────────────

        T0: Alice updates profile photo
        T1: Write to primary database
        T2: Track: Alice's last write timestamp = 1001
        T3: Async replication to replicas (slow)
        T4: Alice refreshes page
        T5: Check: Replica timestamp (995) < Alice's write (1001)
        T6: Fallback: Read from primary database
        T7: Alice sees new photo ✓

        T10: Bob views Alice's profile
        T11: Read from replica (timestamp 995)
        T12: Bob sees old photo (acceptable!)
        T13: Eventually replica catches up
        T14: Bob now sees new photo

        KEY: Alice sees own writes immediately
             Others eventually see updates
        ```

        ---

        ### Real-World Examples

        **User Profiles:**
        ```
        Scenario: User updates settings
        ────────────────────────────────
        Without Read-Your-Writes:
        - User changes theme to "dark"
        - User refreshes page
        - Still sees "light" theme (stale replica) ❌
        - User confused, tries again

        With Read-Your-Writes:
        - User changes theme to "dark"
        - User refreshes page
        - Sees "dark" theme immediately ✓
        - Good UX!
        ```

        ---

        ### Implementation Patterns

        | Pattern | How It Works | Pros | Cons |
        |---------|-------------|------|------|
        | **Sticky Sessions** | Route user to same replica | Simple | Single point failure |
        | **Timestamp Tracking** | Track user's write time, check replicas | Reliable | Overhead |
        | **Read from Primary** | Always read from master after write | Guaranteed | High load on primary |

        ---

        ### When to Use

        - User profiles and settings
        - Personal dashboards
        - Account preferences
        - Any user-facing updates

    === "5. Monotonic Reads"

        ### Definition

        **Once a user sees a version, they never see an older version (time doesn't go backwards).**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | No time travel backwards |
        | **Version** | Monotonically increasing |
        | **Scope** | Per-user tracking |
        | **Implementation** | Version tracking, sticky sessions |

        ---

        ### How It Works

        ```
        Comment Thread
        ─────────────────────────────────────────────

        T0: Alice sees comments [A, B, C] (version 100)
        T1: New comment D added (version 101)
        T2: Alice refreshes from slow replica (version 100)
        T3: Check: Replica version (100) < Last seen (100)
        T4: OK, same version, show [A, B, C]

        T5: Alice refreshes again from stale replica (version 99)
        T6: Check: Replica version (99) < Last seen (100) ❌
        T7: Reject! Read from another replica or wait
        T8: Ensure version ≥ 100

        KEY: Alice never sees comments disappear
        ```

        ---

        ### Real-World Examples

        **Social Media Feed:**
        ```
        Without Monotonic Reads:
        ────────────────────────
        User sees feed: [Post1, Post2, Post3, Post4, Post5]
        User scrolls down
        User scrolls up
        Feed now shows: [Post1, Post2, Post3]
        Posts disappeared! ❌ Confusing!

        With Monotonic Reads:
        ─────────────────────
        User always sees ≥ posts they've seen before ✓
        New posts can appear, old posts don't disappear ✓
        ```

        ---

        ### When to Use

        - Social media feeds
        - Comment threads
        - Activity logs
        - Notification lists

    === "6. Weak Consistency"

        ### Definition

        **No consistency guarantees. Best-effort delivery.**

        | Aspect | Description |
        |--------|-------------|
        | **Guarantee** | None |
        | **Ordering** | No guarantees |
        | **Performance** | Highest possible |
        | **Use Case** | Approximate data OK |

        ---

        ### Real-World Examples

        **Live Video Streaming:**
        ```
        Requirement: Low latency > perfect consistency
        ────────────────────────────────────────────
        - Drop frames if network slow (don't wait) ✓
        - Approximate viewer count (don't sync) ✓
        - Eventually consistent chat (don't block) ✓

        Why: Buffering ruins experience
        Better: Slightly degraded quality
        ```

        **Real-Time Analytics:**
        ```
        Dashboard: Website visitors count
        ────────────────────────────────
        Actual: 10,247 visitors
        Display: ~10,200 visitors (approximate)

        Acceptable because:
        - Approximate count sufficient for decisions
        - Exact count not business-critical
        - Real-time speed matters more
        ```

        ---

        ### When to Use

        - Live streaming (video, audio)
        - Real-time leaderboards
        - Approximate counters
        - Best-effort notifications

=== "🛠️ Implementation Patterns"

    ## How Systems Achieve Consistency

    ### Replication Strategies

    | Strategy | Mechanism | Consistency | Latency | Availability |
    |----------|-----------|-------------|---------|--------------|
    | **Synchronous** | Wait for all replicas | Strong | High (200-500ms) | Low (any failure = unavailable) |
    | **Asynchronous** | Write local, sync later | Eventual | Low (5-50ms) | High (always available) |
    | **Quorum** | Wait for majority | Tunable | Medium (50-200ms) | Medium (majority must be up) |
    | **Primary-Backup** | Write primary, async to backups | Read-your-writes | Medium (20-100ms) | Medium (primary is SPOF) |

    ---

    ### Quorum-Based Consistency

    **The most flexible approach** - tune consistency vs. performance:

    ```
    Configuration:
    ──────────────
    N = Total replicas (e.g., 5)
    W = Write quorum  (how many must confirm write)
    R = Read quorum   (how many must agree on read)

    Strong Consistency: W + R > N
    ──────────────────────────────
    Example: N=5, W=3, R=3 → 3+3=6 > 5 ✓
    - Write to 3 replicas
    - Read from 3 replicas
    - At least 1 replica overlaps
    - Guaranteed to see latest write

    Eventual Consistency: W + R ≤ N
    ─────────────────────────────────
    Example: N=5, W=2, R=2 → 2+2=4 ≤ 5
    - Write to 2 replicas (fast!)
    - Read from 2 replicas (fast!)
    - No overlap guarantee
    - May see stale data

    Tuning Examples:
    ────────────────
    Fast writes: W=1, R=5  (optimize for writes)
    Fast reads:  W=5, R=1  (optimize for reads)
    Balanced:    W=3, R=3  (balanced performance)
    ```

    **Used by:** Cassandra, DynamoDB, Riak (all support tunable consistency)

    ---

    ### Consistency Across Microservices

    **Saga Pattern** - maintain consistency without distributed transactions:

    === "Orchestration"

        ```
        Centralized Coordinator
        ───────────────────────────────────────

        Order Saga:
        1. Reserve inventory → Success ✓
        2. Charge payment → Success ✓
        3. Create shipment → Success ✓
        Result: Order complete

        Order Saga (with failure):
        1. Reserve inventory → Success ✓
        2. Charge payment → FAILED ✗
        Compensation:
        2a. Release inventory
        Result: Order cancelled

        Pros: Clear flow, easy to debug
        Cons: Single point of failure (coordinator)
        ```

    === "Choreography"

        ```
        Event-Driven (No Central Coordinator)
        ─────────────────────────────────────

        Order Flow:
        1. OrderService publishes "ORDER_CREATED"
        2. InventoryService listens, reserves stock, publishes "INVENTORY_RESERVED"
        3. PaymentService listens, charges card, publishes "PAYMENT_COMPLETED"
        4. ShippingService listens, creates shipment

        Failure Flow:
        1. OrderService publishes "ORDER_CREATED"
        2. InventoryService listens, reserves stock, publishes "INVENTORY_RESERVED"
        3. PaymentService listens, charge fails, publishes "PAYMENT_FAILED"
        4. InventoryService listens to failure, releases stock

        Pros: No single point of failure, scalable
        Cons: Complex to debug, eventual consistency
        ```

=== "💡 Interview Tips"

    ## Common Interview Questions

    **Q1: "Explain the difference between strong and eventual consistency"**

    **Good Answer:**
    ```
    Strong Consistency:
    ─────────────────
    - Every read returns the most recent write
    - System behaves like single database
    - Example: Banking - can't show wrong balance
    - Trade-off: Higher latency (200-500ms), lower availability

    Eventual Consistency:
    ───────────────────
    - Replicas converge over time (usually seconds)
    - Temporary inconsistency acceptable
    - Example: Social media - seeing post 10s late is OK
    - Trade-off: Low latency (5-50ms), high availability

    Real-World Decision:
    ──────────────────
    - Banking transfers: Strong (money must be accurate)
    - Product catalog: Eventual (stale price OK for seconds)
    - Inventory: Strong (can't oversell)
    - User profiles: Eventual (avatar update can lag)
    - Payment processing: Strong (no double-charging)

    Key insight: Different parts of system need different models
    ```

    ---

    **Q2: "How do you handle conflicts in eventual consistency?"**

    **Good Answer:**
    ```
    Three Main Strategies:

    1. Last-Write-Wins (LWW)
       - Use timestamp to pick winner
       - Simple but loses data
       - Example: User settings (last update wins)

    2. Application-Defined Merge
       - Custom logic based on data semantics
       - No data loss if merge makes sense
       - Example: Shopping cart (union of items)

    3. Conflict Detection + Manual Resolution
       - Detect with vector clocks
       - Present conflict to user or admin
       - Example: Collaborative editing (Google Docs)

    Decision factors:
    - Can data be merged? (cart: yes, email: no)
    - Is data loss acceptable? (likes: yes, payment: no)
    - User intervention possible? (docs: yes, backend: no)
    ```

    ---

    **Q3: "What is read-your-writes consistency and when do you need it?"**

    **Good Answer:**
    ```
    Definition:
    ─────────
    A user always sees their own updates immediately
    (but may not see others' updates right away)

    Real-World Scenario:
    ──────────────────
    User uploads profile photo → refreshes page → sees new photo
    Without this: User sees old photo (stale replica) → confused!
    With this: User always sees their own changes ✓

    Implementation:
    ─────────────
    1. Track user's last write timestamp
    2. On read, check if replica is up-to-date
    3. If stale, read from primary database

    When to Use:
    ──────────
    - User profiles and settings
    - Account preferences
    - Personal dashboards
    - Any user-facing updates

    Note: Weaker than strong consistency
    (only guarantees for the writing user)
    ```

    ---

    **Q4: "Explain quorum-based consistency"**

    **Good Answer:**
    ```
    Quorum = Majority Vote System

    Configuration:
    ────────────
    N = 5 replicas total
    W = 3 replicas must confirm write
    R = 3 replicas must agree on read

    Strong Consistency: W + R > N
    ───────────────────────────
    Example: W=3, R=3 → 3+3=6 > 5 ✓
    At least 1 replica overlaps between read and write
    Guarantees you see latest write

    Eventual Consistency: W + R ≤ N
    ─────────────────────────────
    Example: W=2, R=2 → 2+2=4 ≤ 5
    No overlap guarantee
    Faster but may read stale data

    Tuning for Different Workloads:
    ─────────────────────────────
    Write-heavy: W=1, R=5 (fast writes)
    Read-heavy:  W=5, R=1 (fast reads)
    Balanced:    W=3, R=3 (equal)

    Systems: Cassandra, DynamoDB, Riak
    ```

    ---

    **Q5: "When would you choose causal consistency?"**

    **Good Answer:**
    ```
    Causal Consistency = Preserve cause-effect relationships

    Perfect For:
    ──────────
    1. Chat Systems
       - Reply must appear after original message
       - But unrelated messages can be reordered

    2. Social Media
       - Comment must appear after post
       - But likes can be reordered

    3. Collaborative Editing
       - Dependent edits must be ordered
       - But independent edits can be concurrent

    Why Not Strong?
    ─────────────
    Strong consistency: ALL operations ordered (slow)
    Causal consistency: Only RELATED operations ordered (faster)

    Example:
    ──────
    Chat thread:
    - Message 1: "What time is the meeting?"
    - Message 2: "3pm" (reply to #1)
    - Message 3: "Who won the game?" (unrelated)

    Causal: #2 always after #1, #3 can be anywhere ✓
    Strong: All messages globally ordered (unnecessary overhead) ✗

    Trade-off: Complex to implement but natural for interactions
    ```

    ---

    ## Interview Cheat Sheet

    **Quick Comparison:**

    | Model | Latency | Complexity | When to Use |
    |-------|---------|------------|-------------|
    | **Strong** | 200-500ms | Low | Banking, inventory, bookings |
    | **Eventual** | 5-50ms | Medium | Social media, catalogs, CDN |
    | **Causal** | 50-100ms | High | Chat, collaborative editing |
    | **Read-Your-Writes** | 10-50ms | Low | User profiles, settings |
    | **Monotonic Reads** | 10-50ms | Low | Feeds, comment threads |
    | **Weak** | <5ms | Lowest | Live streams, analytics |

    **Conflict Resolution:**
    - **Last-Write-Wins**: Simple, loses data, use for settings
    - **Merge**: No data loss, complex, use for carts
    - **Manual**: User decides, use for docs

    **Replication:**
    - **Sync**: Strong consistency, high latency
    - **Async**: Eventual consistency, low latency
    - **Quorum**: Tunable (W + R > N = strong)

    **CAP Theorem Connection:**
    - **CP Systems**: Strong consistency, sacrifice availability
    - **AP Systems**: Eventual consistency, sacrifice consistency

    ---

    ## Practice Problems

    **1. Design a distributed counter for website views**
    - Model: Eventual consistency (approximate count OK)
    - Reason: High write volume, exactness not critical
    - Implementation: Local counters, periodic sync

    **2. Design a seat booking system (airline/theater)**
    - Model: Strong consistency (no double-booking)
    - Reason: Overbooking = refunds + angry customers
    - Implementation: Distributed locks, quorum writes

    **3. Design a collaborative document editor**
    - Model: Causal consistency (preserve edit dependencies)
    - Reason: Dependent edits must be ordered
    - Implementation: Operational Transform or CRDTs

    **4. Design a social media "like" button**
    - Model: Eventual consistency (approximate count OK)
    - Reason: Like count accuracy not critical
    - Implementation: Async aggregation, eventual sync

    **5. Design a video conferencing system**
    - Model: Weak consistency (speed > perfect delivery)
    - Reason: Low latency critical, dropped frames OK
    - Implementation: Best-effort UDP, no retries

=== "⚠️ Common Mistakes"

    ## Misconceptions

    | Mistake | Reality | Why It Matters |
    |---------|---------|----------------|
    | **"Strong = always slow"** | Strong can be fast within datacenter (5-10ms) | Oversimplification |
    | **"Eventual = eventually"** | Usually seconds, not hours. Define SLA! | Sets wrong expectations |
    | **"One model fits all"** | Different features need different models | Over-engineering or under-engineering |
    | **"Consistency is binary"** | It's a spectrum with many levels | Misses optimization opportunities |
    | **"Read-your-writes = strong"** | Much weaker (only per-user) | Confusing guarantees |
    | **"LWW always works"** | Loses data in conflicts | Silent data loss risk |
    | **"Eventual = no conflicts"** | Still need resolution strategy | Production surprises |

    ---

    ## Design Pitfalls

    | Pitfall | Problem | Solution |
    |---------|---------|----------|
    | **No conflict resolution** | System breaks on concurrent writes | Define strategy upfront (LWW, merge, manual) |
    | **Ignoring network delays** | Assumes instant replication | Design for lag (seconds to minutes) |
    | **No consistency SLA** | "Eventual" undefined | Specify window: "99% within 100ms" |
    | **Wrong model for use case** | Banking with eventual, social with strong | Match model to business requirements |
    | **Assuming synchronized clocks** | LWW fails with clock skew | Use logical clocks (version numbers) |
    | **No monitoring** | Can't detect violations | Track replication lag, conflict rate |
    | **No testing** | Production surprises | Test with network partitions, delays |

    ---

    ## Interview Red Flags

    **Avoid Saying:**
    - ❌ "We'll use blockchain for consistency" (buzzword without understanding)
    - ❌ "Just use eventual everywhere" (ignores requirements)
    - ❌ "Consistency doesn't matter for social media" (users notice stale data)
    - ❌ "We'll figure out conflicts later" (recipe for disaster)
    - ❌ "Strong consistency means ACID" (confusing terms)

    **Say Instead:**
    - ✅ "For [feature], I'd choose [model] because [business reason]"
    - ✅ "The consistency window is [time], acceptable because [reason]"
    - ✅ "We'll resolve conflicts using [strategy] based on data semantics"
    - ✅ "Different parts need different models: catalog (eventual), inventory (strong)"
    - ✅ "We'll define SLAs: 99% of reads consistent within 100ms"

---

## 🎯 Key Takeaways

**The 10 Rules of Data Consistency:**

1. **Consistency is a spectrum** - Not binary, many levels between strong and eventual

2. **Choose the weakest model that works** - Stronger = more expensive, weaker = faster

3. **Different features, different models** - Catalog (eventual) + Inventory (strong) = optimal

4. **Define the consistency window** - "Eventual" means what? 100ms? 10s? 1 minute? Be specific!

5. **Plan for conflicts** - Eventual consistency requires a resolution strategy (LWW, merge, manual)

6. **Monitor consistency** - Track replication lag, conflict rate, consistency violations

7. **Test with failures** - Network partitions, delays, clock skew, concurrent writes

8. **Read-your-writes ≠ strong** - Much weaker guarantee (only per-user visibility)

9. **Quorum is tunable** - W + R > N for strong, W + R ≤ N for eventual

10. **Document your choices** - Explain why each feature uses its consistency model

---

## 📚 Further Reading

**Master these related concepts:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **CAP Theorem** | Consistency vs Availability trade-off | [CAP Theorem →](cap-theorem.md) |
| **Replication** | How to sync data across nodes | [Database Replication →](../databases/replication.md) |
| **Consensus** | How nodes agree on values | [Consensus Algorithms →](../distributed-systems/consensus.md) |
| **Transactions** | ACID guarantees in distributed systems | [Distributed Transactions →](../databases/transactions.md) |

**Practice with real systems:**
- [Design WhatsApp](../problems/whatsapp.md) - Message ordering (causal consistency)
- [Design Instagram](../problems/instagram.md) - Feed consistency (eventual)
- [Design Booking System](../problems/booking-system.md) - Strong consistency for seats

---

**Master consistency models and build reliable distributed systems! 🚀**
