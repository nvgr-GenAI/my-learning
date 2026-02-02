# Design Google Docs (Real-time Collaboration)

A collaborative document editing platform where multiple users can simultaneously edit documents with real-time synchronization, conflict resolution, version history, and rich text formatting.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 2B users, 100M concurrent editors, 1B documents, 10M concurrent sessions |
| **Key Challenges** | Operational Transformation (OT), conflict resolution, real-time sync, offline support |
| **Core Concepts** | OT algorithm, CRDT, WebSocket, diff-match-patch, vector clocks, eventual consistency |
| **Companies** | Google, Microsoft (Office 365), Notion, Dropbox Paper, Confluence |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Real-time Editing** | Multiple users edit simultaneously with instant sync | P0 (Must have) |
    | **Conflict Resolution** | Automatic merge of concurrent edits | P0 (Must have) |
    | **Collaborative Cursors** | Show other users' cursors and selections | P0 (Must have) |
    | **Version History** | Track all changes, restore previous versions | P0 (Must have) |
    | **Rich Text Formatting** | Bold, italic, headings, lists, links, images | P0 (Must have) |
    | **Comments & Suggestions** | In-line comments, suggestion mode | P1 (Should have) |
    | **Offline Support** | Edit offline, sync when back online | P1 (Should have) |
    | **Sharing & Permissions** | Share with view/comment/edit permissions | P1 (Should have) |
    | **Auto-save** | Save changes automatically every few seconds | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Advanced formatting (custom fonts, complex tables)
    - Voice typing / dictation
    - Add-ons / extensions
    - Translation services
    - Export to multiple formats (PDF, Word)
    - Document templates

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Edit Sync)** | < 100ms p95 | Users expect instant feedback for collaborative editing |
    | **Availability** | 99.95% uptime | Critical productivity tool, minimal downtime acceptable |
    | **Consistency** | Eventual consistency | Brief conflicts acceptable if resolved automatically |
    | **Conflict Resolution** | 100% conflict-free | No data loss, all edits preserved |
    | **Offline Support** | Full editing capability | Must work without internet, sync later |
    | **Scalability** | 100 concurrent editors/doc | Support large team collaborations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total Users: 2B
    Daily Active Users (DAU): 400M (20% of total)
    Monthly Active Users (MAU): 800M (40% of total)

    Document edits:
    - Active editors: 100M concurrent (during peak hours)
    - Edits per user per session: ~500 edits/hour
    - Edit operations: 100M × 500 / 3600 = ~13.9M edits/sec peak
    - Average: ~7M edits/sec

    Document views (read-only):
    - Viewers: 200M concurrent (peak)
    - View requests: 200M / 60 = ~3.3M views/sec

    Document creation:
    - New docs: 400M DAU × 0.5 docs/day = 200M docs/day
    - Creation QPS: 200M / 86,400 = ~2,300 docs/sec

    Collaboration sessions:
    - Concurrent sessions: 10M (multiple users per document)
    - Session establishment: 10M / 3600 = ~2,800 sessions/sec

    Total Read QPS: ~3.3M (views)
    Total Write QPS: ~7M (edits + auto-saves)
    Read/Write ratio: 1:2 (write-heavy during editing)
    ```

    ### Storage Estimates

    ```
    Document storage:
    - Average document: 50 KB (text + formatting)
    - 1B documents × 50 KB = 50 TB

    Version history:
    - Average versions per doc: 100 versions
    - Incremental storage (delta only): 5 KB per version
    - 1B docs × 100 versions × 5 KB = 500 TB

    Operational Transform log:
    - Operations per doc: ~10,000 operations
    - Operation size: 100 bytes (type, position, content, timestamp)
    - 1B docs × 10,000 ops × 100 bytes = 1 PB

    Media (images, embedded content):
    - 20% of docs have media (~200M docs)
    - Average size: 5 MB per document
    - 200M × 5 MB = 1 PB

    User data:
    - 2B users × 10 KB = 20 TB

    Total: 50 TB (docs) + 500 TB (versions) + 1 PB (OT log) + 1 PB (media) + 20 TB (users) ≈ 2 PB
    ```

    ### Bandwidth Estimates

    ```
    Edit ingress:
    - 7M edits/sec × 1 KB (edit operation) = 7 GB/sec ≈ 56 Gbps

    Edit broadcast (to collaborators):
    - Average 5 collaborators per active document
    - 7M edits/sec × 5 × 1 KB = 35 GB/sec ≈ 280 Gbps

    Document loads:
    - 3.3M views/sec × 50 KB = 165 GB/sec ≈ 1.3 Tbps

    Auto-save:
    - 100M active editors, save every 60s
    - 100M / 60 × 50 KB = 83 GB/sec ≈ 664 Gbps

    Total ingress: ~720 Gbps
    Total egress: ~1.6 Tbps (CDN critical)
    ```

    ### Memory Estimates (Caching)

    ```
    Active documents (in-memory):
    - 10M concurrent sessions × 50 KB = 500 GB
    - Operational Transform state per doc: 10 KB
    - 10M × 10 KB = 100 GB

    WebSocket connections:
    - 100M concurrent connections × 10 KB (state) = 1 TB

    User sessions:
    - 100M active editors × 10 KB = 1 TB

    Document cache (hot documents):
    - 100M hot documents × 50 KB = 5 TB
    - Cache 20% most accessed: 1 TB

    Total cache: 500 GB + 100 GB + 1 TB + 1 TB + 1 TB ≈ 3.6 TB
    ```

    ---

    ## Key Assumptions

    1. Average document size: 50 KB (10-20 pages of text)
    2. 100M concurrent editors during peak hours
    3. Average 5 collaborators per active document
    4. Real-time sync critical (< 100ms latency)
    5. Eventual consistency acceptable (conflicts resolved automatically)
    6. Most edits are small (character insertion/deletion)
    7. Version history retained for 30 days (full), 1 year (daily snapshots)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Operational Transformation (OT):** Resolve concurrent edits deterministically
    2. **Real-time synchronization:** WebSocket for instant edit propagation
    3. **Optimistic updates:** Apply edits locally, sync in background
    4. **Eventual consistency:** All clients converge to same document state
    5. **Offline-first design:** Full editing capability without internet

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
            Desktop[Desktop App]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets]
            LB[Load Balancer<br/>WebSocket aware]
        end

        subgraph "API Layer"
            Doc_API[Document Service<br/>CRUD operations]
            Collab_API[Collaboration Service<br/>Real-time sync]
            Version_API[Version Service<br/>History & restore]
            Comment_API[Comment Service<br/>Comments/suggestions]
            Share_API[Sharing Service<br/>Permissions]
        end

        subgraph "Real-time Layer"
            WS_Gateway[WebSocket Gateway<br/>Connection management]
            OT_Engine[OT Engine<br/>Transform operations]
            Presence[Presence Service<br/>Active users/cursors]
        end

        subgraph "Processing Layer"
            OT_Worker[OT Worker<br/>Apply operations]
            Snapshot_Worker[Snapshot Worker<br/>Create checkpoints]
            Conflict_Resolver[Conflict Resolver<br/>Merge offline edits]
            Search_Indexer[Search Indexer<br/>Full-text search]
        end

        subgraph "Caching"
            Redis_Doc[Redis<br/>Document cache]
            Redis_OT[Redis<br/>OT state]
            Redis_Session[Redis<br/>Session data]
            Redis_Presence[Redis<br/>Presence data]
        end

        subgraph "Storage"
            Doc_DB[(Document DB<br/>MongoDB<br/>Sharded)]
            OT_DB[(Operation Log<br/>Cassandra<br/>Append-only)]
            Version_DB[(Version DB<br/>S3<br/>Snapshots)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Search_DB[(Elasticsearch<br/>Full-text search)]
            Blob_Storage[Object Storage<br/>S3<br/>Media files]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Operation stream]
        end

        Mobile --> CDN
        Web --> CDN
        Desktop --> CDN
        Mobile --> LB
        Web --> LB
        Desktop --> LB

        LB --> Doc_API
        LB --> Collab_API
        LB --> Version_API
        LB --> Comment_API
        LB --> Share_API
        LB --> WS_Gateway

        Doc_API --> Doc_DB
        Doc_API --> Redis_Doc
        Doc_API --> Kafka

        Collab_API --> WS_Gateway
        Collab_API --> OT_Engine
        Collab_API --> Kafka

        WS_Gateway --> Presence
        WS_Gateway --> Redis_Session
        WS_Gateway --> Redis_Presence

        OT_Engine --> Redis_OT
        OT_Engine --> OT_DB

        Kafka --> OT_Worker
        Kafka --> Snapshot_Worker
        Kafka --> Conflict_Resolver
        Kafka --> Search_Indexer

        OT_Worker --> OT_DB
        OT_Worker --> Doc_DB

        Snapshot_Worker --> Version_DB
        Snapshot_Worker --> Doc_DB

        Search_Indexer --> Search_DB

        Version_API --> Version_DB
        Comment_API --> Doc_DB
        Share_API --> User_DB

        Presence --> Redis_Presence

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Doc fill:#fff4e1
        style Redis_OT fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Redis_Presence fill:#fff4e1
        style Doc_DB fill:#ffe1e1
        style OT_DB fill:#e8eaf6
        style Version_DB fill:#f3e5f5
        style User_DB fill:#ffe1e1
        style Search_DB fill:#e8eaf6
        style Blob_Storage fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Operational Transformation** | Deterministic conflict resolution, maintains intent | CRDT (more complex for rich text), pessimistic locking (poor UX) |
    | **WebSocket** | Real-time bidirectional communication (<100ms sync) | Polling (high latency, wasteful), Server-Sent Events (one-way only) |
    | **Cassandra (OT Log)** | Append-only operations, high write throughput (7M ops/sec) | MySQL (write bottleneck), MongoDB (ordering guarantees) |
    | **MongoDB (Documents)** | Flexible schema, document-oriented, fast reads/writes | PostgreSQL (rigid schema), Cassandra (complex queries) |
    | **Redis** | In-memory state for active documents, presence data | No cache (DB can't handle 7M write QPS), Memcached (limited data structures) |
    | **Kafka** | Reliable operation streaming, replay capability | RabbitMQ (can't handle throughput), direct processing (no replay) |

    **Key Trade-off:** We chose **eventual consistency over strong consistency**. Brief editing conflicts acceptable as long as they're resolved automatically without data loss.

    ---

    ## API Design

    ### 1. Open Document

    **Request:**
    ```http
    GET /api/v1/documents/{doc_id}?version=latest
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "document_id": "doc_abc123",
      "title": "Q4 Planning Document",
      "content": {
        "ops": [
          { "insert": "Hello World\n" },
          { "insert": "Bold text", "attributes": { "bold": true } }
        ]
      },
      "version": 1523,
      "created_at": "2026-01-15T10:00:00Z",
      "updated_at": "2026-02-02T14:30:00Z",
      "owner": {
        "user_id": "user123",
        "name": "John Doe",
        "email": "john@example.com"
      },
      "collaborators": [
        {
          "user_id": "user456",
          "name": "Jane Smith",
          "permission": "edit",
          "status": "online"
        }
      ],
      "websocket_url": "wss://collab.docs.com/doc_abc123?token=<ws_token>"
    }
    ```

    **Design Notes:**

    - Return current document version (for OT)
    - Include WebSocket URL for real-time sync
    - Show active collaborators and their status
    - Content in Quill Delta format (standard for rich text)

    ---

    ### 2. Apply Edit Operation

    **WebSocket Message (Client → Server):**
    ```json
    {
      "type": "operation",
      "document_id": "doc_abc123",
      "operation": {
        "op_id": "op_789",
        "user_id": "user123",
        "base_version": 1523,
        "ops": [
          {
            "retain": 12,
            "attributes": { "bold": true }
          }
        ],
        "timestamp": "2026-02-02T14:30:05.123Z"
      }
    }
    ```

    **WebSocket Response (Server → Client):**
    ```json
    {
      "type": "operation_ack",
      "op_id": "op_789",
      "document_id": "doc_abc123",
      "new_version": 1524,
      "status": "applied",
      "timestamp": "2026-02-02T14:30:05.156Z"
    }
    ```

    **Broadcast to Other Clients:**
    ```json
    {
      "type": "remote_operation",
      "document_id": "doc_abc123",
      "operation": {
        "op_id": "op_789",
        "user_id": "user123",
        "version": 1524,
        "ops": [
          {
            "retain": 12,
            "attributes": { "bold": true }
          }
        ]
      },
      "user": {
        "user_id": "user123",
        "name": "John Doe"
      }
    }
    ```

    **Design Notes:**

    - Operations include base version (for OT transformation)
    - Server acknowledges with new version
    - Broadcast transformed operation to other clients
    - Client applies local operation immediately (optimistic update)

    ---

    ### 3. Cursor/Selection Update

    **WebSocket Message:**
    ```json
    {
      "type": "cursor",
      "document_id": "doc_abc123",
      "cursor": {
        "user_id": "user123",
        "index": 42,
        "length": 0,
        "color": "#FF5722"
      }
    }
    ```

    ---

    ### 4. Get Version History

    **Request:**
    ```http
    GET /api/v1/documents/{doc_id}/versions?limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "versions": [
        {
          "version": 1524,
          "created_at": "2026-02-02T14:30:05Z",
          "user": {
            "user_id": "user123",
            "name": "John Doe"
          },
          "change_summary": "Added bold formatting to 'Hello World'",
          "operations_count": 1
        },
        // ... more versions
      ],
      "has_more": true
    }
    ```

    ---

    ## Database Schema

    ### Documents (MongoDB)

    ```javascript
    // documents collection
    {
      _id: "doc_abc123",
      title: "Q4 Planning Document",
      owner_id: "user123",

      // Current document state (Quill Delta format)
      content: {
        ops: [
          { insert: "Hello World\n" },
          { insert: "Bold text", attributes: { bold: true } }
        ]
      },

      // Version tracking
      current_version: 1524,

      // Metadata
      created_at: ISODate("2026-01-15T10:00:00Z"),
      updated_at: ISODate("2026-02-02T14:30:05Z"),

      // Collaborators
      collaborators: [
        {
          user_id: "user456",
          permission: "edit",  // view, comment, edit
          added_at: ISODate("2026-01-16T12:00:00Z")
        }
      ],

      // Settings
      settings: {
        public: false,
        allow_comments: true,
        allow_suggestions: true
      },

      // Indexing
      tags: ["planning", "q4"],
      folder_id: "folder_xyz"
    }

    // Indexes
    db.documents.createIndex({ "owner_id": 1, "updated_at": -1 })
    db.documents.createIndex({ "collaborators.user_id": 1 })
    db.documents.createIndex({ "title": "text", "tags": "text" })
    ```

    **Why MongoDB:**

    - **Flexible schema:** Content structure varies (rich text, formatting)
    - **Document-oriented:** Natural fit for document storage
    - **Fast reads/writes:** Handles 3.3M views/sec
    - **Sharding:** Horizontal scaling by document_id

    ---

    ### Operations Log (Cassandra)

    ```sql
    -- operations table (append-only log)
    CREATE TABLE operations (
        document_id TEXT,
        version BIGINT,
        op_id TEXT,
        user_id TEXT,
        operation TEXT,  -- JSON-serialized operation
        timestamp TIMESTAMP,
        PRIMARY KEY (document_id, version)
    ) WITH CLUSTERING ORDER BY (version ASC);

    -- User operations (for conflict resolution)
    CREATE TABLE user_operations (
        user_id TEXT,
        document_id TEXT,
        timestamp TIMESTAMP,
        version BIGINT,
        op_id TEXT,
        PRIMARY KEY (user_id, document_id, timestamp)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);

    -- Pending operations (offline edits)
    CREATE TABLE pending_operations (
        document_id TEXT,
        user_id TEXT,
        client_version BIGINT,
        op_id TEXT,
        operation TEXT,
        timestamp TIMESTAMP,
        synced BOOLEAN,
        PRIMARY KEY (document_id, user_id, client_version)
    );
    ```

    **Why Cassandra:**

    - **Write-optimized:** 7M write ops/sec (append-only)
    - **Time-series data:** Operations ordered by version
    - **Linear scalability:** Add nodes for more throughput
    - **No single point of failure:** Multi-master replication

    ---

    ### Version Snapshots (S3)

    ```
    s3://docs-versions/
    ├── doc_abc123/
    │   ├── v1000.json      # Snapshot at version 1000
    │   ├── v2000.json      # Snapshot at version 2000
    │   ├── v3000.json      # Every 1000 versions
    │   └── daily/
    │       ├── 2026-01-15.json
    │       └── 2026-01-16.json
    ```

    **Snapshot strategy:**

    - **Checkpoint every 1000 operations:** Fast restore (replay from nearest checkpoint)
    - **Daily snapshots:** Version history UI
    - **Compressed JSON:** Reduce storage costs

    ---

    ### Users (PostgreSQL)

    ```sql
    -- users table (sharded by user_id)
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(100),
        profile_pic_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_active TIMESTAMP,
        storage_used BIGINT DEFAULT 0,
        storage_limit BIGINT DEFAULT 15_000_000_000,  -- 15 GB
        INDEX idx_email (email)
    ) PARTITION BY HASH (user_id);

    -- sharing permissions
    CREATE TABLE document_shares (
        share_id BIGINT PRIMARY KEY,
        document_id TEXT NOT NULL,
        user_id BIGINT NOT NULL,
        permission VARCHAR(10),  -- view, comment, edit
        shared_by BIGINT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_document (document_id),
        INDEX idx_user (user_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Real-time Edit Flow

    ```mermaid
    sequenceDiagram
        participant Client1 as Client 1 (Editor)
        participant Client2 as Client 2 (Editor)
        participant WS_Gateway as WebSocket Gateway
        participant OT_Engine as OT Engine
        participant Redis as Redis (OT State)
        participant Kafka as Kafka
        participant OT_Worker as OT Worker
        participant Doc_DB as Document DB

        Note over Client1: User types "Hello"
        Client1->>Client1: Apply operation locally (optimistic)
        Client1->>WS_Gateway: Send operation (base_version: 100)

        WS_Gateway->>OT_Engine: Transform operation
        OT_Engine->>Redis: Get document state (version 100)
        Redis-->>OT_Engine: Current version: 101 (conflict!)

        OT_Engine->>OT_Engine: Transform operation (100→101)
        OT_Engine->>Redis: Update state (version 102)
        OT_Engine-->>WS_Gateway: Transformed operation (version 102)

        WS_Gateway-->>Client1: ACK (version 102)
        WS_Gateway->>Client2: Broadcast operation

        Client2->>Client2: Apply remote operation

        WS_Gateway->>Kafka: Publish operation event
        Kafka->>OT_Worker: Process operation
        OT_Worker->>Doc_DB: Persist to database (async)
    ```

    **Flow Explanation:**

    1. **Local application** - Client applies edit immediately (optimistic)
    2. **Send to server** - Operation includes base version
    3. **Transformation** - OT Engine transforms if there are conflicts
    4. **Broadcast** - Send to all collaborators via WebSocket
    5. **Persist** - Async save to database (eventual consistency)

    ---

    ### Document Load Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Doc_API
        participant Redis
        participant Doc_DB
        participant OT_DB
        participant WS_Gateway
        participant Presence

        Client->>Doc_API: GET /documents/{doc_id}
        Doc_API->>Doc_API: Authenticate & authorize

        Doc_API->>Redis: GET doc:{doc_id}
        alt Cache HIT
            Redis-->>Doc_API: Cached document
        else Cache MISS
            Redis-->>Doc_API: null
            Doc_API->>Doc_DB: Query document
            Doc_DB-->>Doc_API: Document data
            Doc_API->>Redis: SET doc:{doc_id} (TTL: 300s)
        end

        Doc_API-->>Client: Document + WebSocket URL

        Client->>WS_Gateway: Connect WebSocket
        WS_Gateway->>Presence: Register user presence
        WS_Gateway->>Redis: Subscribe to doc:{doc_id} channel
        WS_Gateway-->>Client: Connection established

        WS_Gateway->>Client: Send current collaborators
        WS_Gateway->>Presence: Broadcast user joined
    ```

    ---

    ### Offline Sync Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Local_Storage as Local Storage
        participant WS_Gateway as WebSocket Gateway
        participant Conflict_Resolver as Conflict Resolver
        participant OT_Engine as OT Engine
        participant Doc_DB as Document DB

        Note over Client: User goes offline
        Client->>Client: Detect offline mode
        Client->>Local_Storage: Store operations locally

        Note over Client: User makes 50 edits offline
        loop Each edit
            Client->>Client: Apply operation locally
            Client->>Local_Storage: Queue operation
        end

        Note over Client: User comes back online
        Client->>WS_Gateway: Reconnect WebSocket
        WS_Gateway-->>Client: Current document version: 250

        Note over Client: Client's version: 200
        Client->>WS_Gateway: Send queued operations (versions 201-250)

        WS_Gateway->>Conflict_Resolver: Resolve conflicts
        Conflict_Resolver->>OT_Engine: Transform operations (200→250)
        OT_Engine->>OT_Engine: Apply transformations
        OT_Engine-->>WS_Gateway: Transformed operations

        WS_Gateway->>Doc_DB: Persist merged operations
        WS_Gateway-->>Client: Sync complete (new version: 300)
        WS_Gateway->>Client: Broadcast operations to other users
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Google Docs subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Operational Transformation** | How to merge concurrent edits without conflicts? | OT algorithm with transformation functions |
    | **CRDT Alternative** | When to use CRDT instead of OT? | CRDT for better offline support, OT for rich text |
    | **Real-time Sync** | How to sync edits in <100ms? | WebSocket + Redis pub/sub + optimistic updates |
    | **Version History** | How to store infinite versions efficiently? | Snapshot + delta compression + S3 archival |

    ---

    === "🔄 Operational Transformation (OT)"

        ## The Challenge

        **Problem:** Two users edit simultaneously. How to merge their changes without conflicts or data loss?

        **Example scenario:**

        ```
        Initial document: "Hello World"
                          012345678910

        User A (position 6): Insert "Beautiful " → "Hello Beautiful World"
        User B (position 6): Insert "Cruel " → "Hello Cruel World"

        Without OT: 💥 Conflict! Which edit wins?
        With OT: ✅ "Hello Beautiful Cruel World" (both edits preserved)
        ```

        **OT guarantees:**

        1. **Convergence:** All clients converge to identical state
        2. **Causality preservation:** Operations applied in causal order
        3. **Intention preservation:** Original edit intent maintained

        ---

        ## OT Algorithm

        **Core concept:** Transform operations based on concurrent operations.

        **Transformation function:** `transform(op1, op2) → (op1', op2')`

        **Rules:**

        ```
        Given: op1 and op2 are concurrent (same base version)
        After transformation:
        - apply(op1', apply(op2, doc)) = apply(op2', apply(op1, doc))
        ```

        ---

        ## OT Implementation

        ```python
        from dataclasses import dataclass
        from typing import List, Union
        from enum import Enum

        class OpType(Enum):
            """Operation types in OT"""
            INSERT = "insert"
            DELETE = "delete"
            RETAIN = "retain"

        @dataclass
        class Operation:
            """
            Single operation in OT system

            Examples:
            - Insert "Hello" at position 0: Operation(INSERT, 0, "Hello")
            - Delete 5 chars at position 0: Operation(DELETE, 0, 5)
            - Retain 10 chars: Operation(RETAIN, 10)
            """
            type: OpType
            position: int = 0
            content: Union[str, int] = None  # string for INSERT, count for DELETE/RETAIN
            attributes: dict = None  # For formatting (bold, italic, etc.)

        class OperationalTransform:
            """
            Operational Transformation engine for conflict resolution

            Based on Google's Jupiter system and Apache Wave (Google Wave)
            """

            def __init__(self):
                self.document_versions = {}  # doc_id -> version
                self.pending_operations = {}  # doc_id -> list of operations

            def transform(self, op1: Operation, op2: Operation) -> tuple:
                """
                Transform two concurrent operations

                Args:
                    op1: First operation (from client A)
                    op2: Second operation (from client B, concurrent with op1)

                Returns:
                    (op1', op2'): Transformed operations
                """
                # INSERT vs INSERT
                if op1.type == OpType.INSERT and op2.type == OpType.INSERT:
                    return self._transform_insert_insert(op1, op2)

                # INSERT vs DELETE
                elif op1.type == OpType.INSERT and op2.type == OpType.DELETE:
                    return self._transform_insert_delete(op1, op2)

                # DELETE vs INSERT
                elif op1.type == OpType.DELETE and op2.type == OpType.INSERT:
                    op2_prime, op1_prime = self._transform_insert_delete(op2, op1)
                    return op1_prime, op2_prime

                # DELETE vs DELETE
                elif op1.type == OpType.DELETE and op2.type == OpType.DELETE:
                    return self._transform_delete_delete(op1, op2)

                return op1, op2

            def _transform_insert_insert(self, op1: Operation, op2: Operation) -> tuple:
                """
                Transform two concurrent INSERT operations

                Example:
                    Document: "Hello"
                    op1: Insert "A" at position 2 → "HeAllo"
                    op2: Insert "B" at position 2 → "HeBllo"

                    After transform:
                    op1': Insert "A" at position 2
                    op2': Insert "B" at position 3 (shifted right)
                """
                if op1.position < op2.position:
                    # op1 is before op2, op2 shifts right
                    op1_prime = Operation(OpType.INSERT, op1.position, op1.content, op1.attributes)
                    op2_prime = Operation(OpType.INSERT, op2.position + len(op1.content), op2.content, op2.attributes)
                elif op1.position > op2.position:
                    # op2 is before op1, op1 shifts right
                    op1_prime = Operation(OpType.INSERT, op1.position + len(op2.content), op1.content, op1.attributes)
                    op2_prime = Operation(OpType.INSERT, op2.position, op2.content, op2.attributes)
                else:
                    # Same position - use tie-breaker (e.g., user_id or timestamp)
                    # For simplicity, op1 goes first
                    op1_prime = Operation(OpType.INSERT, op1.position, op1.content, op1.attributes)
                    op2_prime = Operation(OpType.INSERT, op2.position + len(op1.content), op2.content, op2.attributes)

                return op1_prime, op2_prime

            def _transform_insert_delete(self, insert_op: Operation, delete_op: Operation) -> tuple:
                """
                Transform INSERT vs DELETE

                Example:
                    Document: "Hello"
                    insert_op: Insert "A" at position 2 → "HeAllo"
                    delete_op: Delete 2 chars at position 1 → "Hlo"

                    After transform:
                    - If insert is after delete range: shift insert left
                    - If insert is before delete range: shift delete right
                    - If insert is inside delete range: insert survives, delete shifts
                """
                delete_start = delete_op.position
                delete_end = delete_op.position + delete_op.content

                if insert_op.position <= delete_start:
                    # Insert is before delete, delete shifts right
                    insert_prime = insert_op
                    delete_prime = Operation(
                        OpType.DELETE,
                        delete_op.position + len(insert_op.content),
                        delete_op.content
                    )
                elif insert_op.position >= delete_end:
                    # Insert is after delete, insert shifts left
                    insert_prime = Operation(
                        OpType.INSERT,
                        insert_op.position - delete_op.content,
                        insert_op.content,
                        insert_op.attributes
                    )
                    delete_prime = delete_op
                else:
                    # Insert is inside delete range
                    # Insert survives at delete_start position
                    insert_prime = Operation(
                        OpType.INSERT,
                        delete_start,
                        insert_op.content,
                        insert_op.attributes
                    )
                    delete_prime = delete_op

                return insert_prime, delete_prime

            def _transform_delete_delete(self, op1: Operation, op2: Operation) -> tuple:
                """
                Transform two concurrent DELETE operations

                Example:
                    Document: "Hello World"
                    op1: Delete 5 chars at position 0 → " World"
                    op2: Delete 5 chars at position 6 → "Hello "

                    After transform: Adjust ranges based on overlap
                """
                op1_start, op1_end = op1.position, op1.position + op1.content
                op2_start, op2_end = op2.position, op2.position + op2.content

                # No overlap
                if op1_end <= op2_start:
                    # op1 is before op2, op2 shifts left
                    op1_prime = op1
                    op2_prime = Operation(OpType.DELETE, op2.position - op1.content, op2.content)
                elif op2_end <= op1_start:
                    # op2 is before op1, op1 shifts left
                    op1_prime = Operation(OpType.DELETE, op1.position - op2.content, op1.content)
                    op2_prime = op2
                else:
                    # Overlapping deletes - calculate intersection
                    intersection_start = max(op1_start, op2_start)
                    intersection_end = min(op1_end, op2_end)
                    intersection_size = intersection_end - intersection_start

                    # Adjust delete sizes
                    op1_prime = Operation(
                        OpType.DELETE,
                        min(op1_start, op2_start),
                        op1.content - intersection_size
                    )
                    op2_prime = Operation(
                        OpType.DELETE,
                        min(op1_start, op2_start),
                        op2.content - intersection_size
                    )

                return op1_prime, op2_prime

            def apply_operation(self, document: str, op: Operation) -> str:
                """
                Apply operation to document

                Args:
                    document: Current document content
                    op: Operation to apply

                Returns:
                    Updated document content
                """
                if op.type == OpType.INSERT:
                    # Insert content at position
                    return document[:op.position] + op.content + document[op.position:]

                elif op.type == OpType.DELETE:
                    # Delete content from position
                    return document[:op.position] + document[op.position + op.content:]

                elif op.type == OpType.RETAIN:
                    # No change to content, used for formatting
                    return document

                return document

            def compose(self, ops: List[Operation]) -> List[Operation]:
                """
                Compose multiple operations into minimal set

                Example:
                    Input: [Insert "A" at 0, Insert "B" at 1, Insert "C" at 2]
                    Output: [Insert "ABC" at 0]
                """
                if not ops:
                    return []

                composed = [ops[0]]

                for op in ops[1:]:
                    last = composed[-1]

                    # Merge consecutive inserts
                    if (last.type == OpType.INSERT and op.type == OpType.INSERT and
                        last.position + len(last.content) == op.position):
                        composed[-1] = Operation(
                            OpType.INSERT,
                            last.position,
                            last.content + op.content,
                            last.attributes
                        )
                    # Merge consecutive deletes
                    elif (last.type == OpType.DELETE and op.type == OpType.DELETE and
                          last.position == op.position):
                        composed[-1] = Operation(
                            OpType.DELETE,
                            last.position,
                            last.content + op.content
                        )
                    else:
                        composed.append(op)

                return composed
        ```

        ---

        ## OT Server Implementation

        ```python
        import asyncio
        import redis.asyncio as redis
        from typing import Dict, List
        import json

        class OTServer:
            """
            Operational Transformation server
            Manages concurrent edits across multiple clients
            """

            def __init__(self):
                self.ot = OperationalTransform()
                self.redis = redis.Redis(host='redis-host')
                self.document_versions = {}  # doc_id -> current version
                self.operation_buffers = {}  # doc_id -> list of pending ops

            async def handle_operation(self, doc_id: str, client_id: str,
                                      operation: Operation, base_version: int):
                """
                Handle incoming operation from client

                Args:
                    doc_id: Document ID
                    client_id: Client who sent operation
                    operation: Operation to apply
                    base_version: Client's base version

                Returns:
                    Transformed operation and new version
                """
                # Get current document version
                current_version = await self._get_version(doc_id)

                # If base version matches current, no transformation needed
                if base_version == current_version:
                    # Apply operation directly
                    transformed_op = operation
                else:
                    # Transform against concurrent operations
                    concurrent_ops = await self._get_operations(doc_id, base_version, current_version)
                    transformed_op = await self._transform_against_concurrent(operation, concurrent_ops)

                # Apply transformed operation
                await self._apply_operation(doc_id, transformed_op)

                # Increment version
                new_version = current_version + 1
                await self._set_version(doc_id, new_version)

                # Store operation in log
                await self._store_operation(doc_id, new_version, client_id, transformed_op)

                # Broadcast to other clients
                await self._broadcast_operation(doc_id, client_id, transformed_op, new_version)

                return transformed_op, new_version

            async def _transform_against_concurrent(self, operation: Operation,
                                                    concurrent_ops: List[Operation]) -> Operation:
                """
                Transform operation against list of concurrent operations

                Args:
                    operation: Operation to transform
                    concurrent_ops: List of concurrent operations

                Returns:
                    Transformed operation
                """
                transformed = operation

                for concurrent_op in concurrent_ops:
                    # Transform against each concurrent operation
                    transformed, _ = self.ot.transform(transformed, concurrent_op)

                return transformed

            async def _get_operations(self, doc_id: str, from_version: int,
                                     to_version: int) -> List[Operation]:
                """
                Get operations between two versions

                Used for transforming against concurrent operations
                """
                ops_data = await self.redis.lrange(f"ops:{doc_id}:{from_version}", 0, -1)
                operations = []

                for op_json in ops_data:
                    op_dict = json.loads(op_json)
                    operations.append(Operation(
                        type=OpType(op_dict['type']),
                        position=op_dict['position'],
                        content=op_dict['content'],
                        attributes=op_dict.get('attributes')
                    ))

                return operations

            async def _apply_operation(self, doc_id: str, operation: Operation):
                """Apply operation to document in Redis"""
                doc_key = f"doc:{doc_id}"
                document = await self.redis.get(doc_key)

                if document:
                    updated_doc = self.ot.apply_operation(document.decode(), operation)
                    await self.redis.set(doc_key, updated_doc)

            async def _store_operation(self, doc_id: str, version: int,
                                      client_id: str, operation: Operation):
                """Store operation in operation log (Cassandra via Kafka)"""
                op_data = {
                    'document_id': doc_id,
                    'version': version,
                    'client_id': client_id,
                    'operation': {
                        'type': operation.type.value,
                        'position': operation.position,
                        'content': operation.content,
                        'attributes': operation.attributes
                    },
                    'timestamp': asyncio.get_event_loop().time()
                }

                # Publish to Kafka for persistence
                await self.redis.publish(f"operation_log", json.dumps(op_data))

            async def _broadcast_operation(self, doc_id: str, sender_id: str,
                                          operation: Operation, version: int):
                """Broadcast operation to all connected clients except sender"""
                message = {
                    'type': 'remote_operation',
                    'document_id': doc_id,
                    'sender_id': sender_id,
                    'version': version,
                    'operation': {
                        'type': operation.type.value,
                        'position': operation.position,
                        'content': operation.content,
                        'attributes': operation.attributes
                    }
                }

                await self.redis.publish(f"doc:{doc_id}", json.dumps(message))

            async def _get_version(self, doc_id: str) -> int:
                """Get current document version"""
                version = await self.redis.get(f"version:{doc_id}")
                return int(version) if version else 0

            async def _set_version(self, doc_id: str, version: int):
                """Set document version"""
                await self.redis.set(f"version:{doc_id}", version)
        ```

        ---

        ## OT Trade-offs

        | Aspect | OT Approach | CRDT Approach |
        |--------|------------|---------------|
        | **Convergence** | Guaranteed with correct transforms | Guaranteed by design |
        | **Complexity** | High (transform functions) | Medium (merge logic) |
        | **Rich text support** | Excellent | Challenging |
        | **Offline edits** | Requires conflict resolution | Natural support |
        | **Bandwidth** | Low (send operations only) | Higher (send full state) |
        | **Implementation** | Proven at scale (Google Docs) | Emerging (Figma, Notion) |

        **Why Google Docs uses OT:**

        - Better for rich text editing (formatting, styles)
        - Lower bandwidth (operations only)
        - Proven at 2B+ users
        - Fine-grained intention preservation

    === "📊 CRDT Alternative"

        ## What are CRDTs?

        **CRDT = Conflict-free Replicated Data Type**

        **Core principle:** Data structure designed to merge automatically without conflicts.

        **Types of CRDTs:**

        1. **G-Counter** (Grow-only counter) - Can only increment
        2. **PN-Counter** (Positive-Negative counter) - Can increment/decrement
        3. **G-Set** (Grow-only set) - Can only add items
        4. **OR-Set** (Observed-Remove set) - Can add/remove items
        5. **LWW-Register** (Last-Write-Wins register) - Single value, last write wins
        6. **RGA** (Replicated Growable Array) - For text editing

        ---

        ## CRDT for Text Editing

        **Problem with naive approach:**

        ```
        Initial: "Hello"
        User A: Insert "A" at position 2 → "HeAllo"
        User B: Insert "B" at position 2 → "HeBllo"
        Merge: ??? → "HeBAllo" or "HeABllo" (non-deterministic!)
        ```

        **Solution: RGA (Replicated Growable Array)**

        Each character has:
        - Unique ID (user_id + timestamp)
        - Position relative to previous character

        ---

        ## CRDT Implementation

        ```python
        from dataclasses import dataclass
        from typing import Optional, Dict, List
        import time

        @dataclass
        class Character:
            """
            Single character in CRDT
            Each character has globally unique ID
            """
            char_id: str  # Format: "{user_id}:{timestamp}:{counter}"
            value: str    # Actual character
            user_id: str
            timestamp: float
            after: Optional[str] = None  # ID of previous character
            deleted: bool = False

            def __lt__(self, other):
                """Compare for sorting (timestamp-based)"""
                return (self.timestamp, self.user_id) < (other.timestamp, other.user_id)

        class CRDT_Text:
            """
            CRDT-based text editor
            Based on RGA (Replicated Growable Array)

            Used by: Figma, Notion (partial), Atom Teletype
            """

            def __init__(self, user_id: str):
                self.user_id = user_id
                self.characters: Dict[str, Character] = {}  # char_id -> Character
                self.counter = 0  # For unique IDs
                self.head_id = "ROOT"  # Start of document

                # Initialize ROOT character
                self.characters[self.head_id] = Character(
                    char_id=self.head_id,
                    value="",
                    user_id="system",
                    timestamp=0.0
                )

            def insert(self, position: int, value: str) -> List[Character]:
                """
                Insert character(s) at position

                Args:
                    position: Position to insert (0-based)
                    value: Character(s) to insert

                Returns:
                    List of Character objects created
                """
                # Find character before insert position
                prev_char_id = self._get_char_id_at_position(position)

                new_chars = []
                for char in value:
                    # Create new character with unique ID
                    char_id = f"{self.user_id}:{time.time()}:{self.counter}"
                    self.counter += 1

                    new_char = Character(
                        char_id=char_id,
                        value=char,
                        user_id=self.user_id,
                        timestamp=time.time(),
                        after=prev_char_id
                    )

                    self.characters[char_id] = new_char
                    new_chars.append(new_char)

                    # Next character comes after this one
                    prev_char_id = char_id

                return new_chars

            def delete(self, position: int) -> Optional[Character]:
                """
                Delete character at position

                Args:
                    position: Position to delete (0-based)

                Returns:
                    Deleted Character (marked as deleted, not removed)
                """
                char_id = self._get_char_id_at_position(position + 1)  # +1 to skip ROOT

                if char_id and char_id != self.head_id:
                    char = self.characters[char_id]
                    char.deleted = True
                    return char

                return None

            def merge(self, remote_chars: List[Character]):
                """
                Merge characters from remote client

                CRDTs guarantee conflict-free merge

                Args:
                    remote_chars: List of characters from other client
                """
                for remote_char in remote_chars:
                    if remote_char.char_id not in self.characters:
                        # New character, add it
                        self.characters[remote_char.char_id] = remote_char
                    else:
                        # Existing character, merge deleted state
                        local_char = self.characters[remote_char.char_id]
                        if remote_char.deleted:
                            local_char.deleted = True

            def to_string(self) -> str:
                """
                Convert CRDT to string representation

                Returns:
                    Current document text
                """
                # Build linked list of characters
                result = []

                # Sort characters by (after, timestamp)
                sorted_chars = sorted(
                    [c for c in self.characters.values() if not c.deleted and c.char_id != self.head_id],
                    key=lambda c: (c.after, c.timestamp, c.user_id)
                )

                # Build string
                current_after = self.head_id
                while True:
                    # Find next character
                    next_chars = [c for c in sorted_chars if c.after == current_after]
                    if not next_chars:
                        break

                    # Use first (earliest timestamp)
                    next_char = next_chars[0]
                    result.append(next_char.value)
                    current_after = next_char.char_id
                    sorted_chars.remove(next_char)

                return ''.join(result)

            def _get_char_id_at_position(self, position: int) -> Optional[str]:
                """
                Get character ID at position

                Args:
                    position: Position (0 = ROOT)

                Returns:
                    Character ID at position
                """
                if position == 0:
                    return self.head_id

                # Build ordered list of character IDs
                sorted_chars = sorted(
                    [c for c in self.characters.values() if not c.deleted],
                    key=lambda c: (c.timestamp, c.user_id)
                )

                if position <= len(sorted_chars):
                    return sorted_chars[position - 1].char_id

                return sorted_chars[-1].char_id if sorted_chars else self.head_id

            def get_state(self) -> List[Dict]:
                """
                Get full CRDT state for synchronization

                Returns:
                    List of character dictionaries
                """
                return [
                    {
                        'char_id': char.char_id,
                        'value': char.value,
                        'user_id': char.user_id,
                        'timestamp': char.timestamp,
                        'after': char.after,
                        'deleted': char.deleted
                    }
                    for char in self.characters.values()
                ]
        ```

        ---

        ## CRDT Example

        ```python
        # Scenario: Two users edit simultaneously

        # User A
        doc_a = CRDT_Text(user_id="alice")
        doc_a.insert(0, "Hello")
        # State: "Hello"

        # User B (same initial state)
        doc_b = CRDT_Text(user_id="bob")
        doc_b.insert(0, "Hello")
        # State: "Hello"

        # Both users edit simultaneously (offline)
        # User A: Insert " World" at end
        ops_a = doc_a.insert(5, " World")
        print(doc_a.to_string())  # "Hello World"

        # User B: Insert " Beautiful" at position 5
        ops_b = doc_b.insert(5, " Beautiful")
        print(doc_b.to_string())  # "Hello Beautiful"

        # Sync: Both users exchange operations
        # User A receives User B's operations
        doc_a.merge(ops_b)
        print(doc_a.to_string())  # "Hello Beautiful World" (deterministic!)

        # User B receives User A's operations
        doc_b.merge(ops_a)
        print(doc_b.to_string())  # "Hello Beautiful World" (same result!)
        ```

        ---

        ## CRDT vs OT Comparison

        | Feature | OT | CRDT |
        |---------|----|----- |
        | **Conflict resolution** | Transform operations | Merge states |
        | **Central server required** | Yes (for ordering) | No (peer-to-peer possible) |
        | **Offline support** | Complex | Natural |
        | **Rich text formatting** | Excellent | Challenging |
        | **Bandwidth** | Low (operations) | Higher (state) |
        | **Complexity** | High | Medium |
        | **Proven scale** | Google Docs (2B users) | Figma (growing) |

        **When to use CRDT:**

        - Peer-to-peer collaboration (no server)
        - Offline-first applications
        - Simple text editing
        - Canvas/drawing apps (Figma)

        **When to use OT:**

        - Rich text editing (Google Docs)
        - Server-mediated collaboration
        - Lower bandwidth requirements
        - Fine-grained intention preservation

    === "⚡ Real-time Synchronization"

        ## The Challenge

        **Problem:** Synchronize edits across 100M concurrent editors in < 100ms.

        **Requirements:**

        - **Low latency:** < 100ms edit propagation
        - **Scalability:** 100M concurrent WebSocket connections
        - **Reliability:** No missed operations
        - **Efficiency:** Minimize bandwidth

        ---

        ## WebSocket Architecture

        **Three-tier architecture:**

        ```
        100M concurrent editors
        ├── 10,000 WebSocket servers (10K connections each)
        │   ├── Server 1: users 0-10K
        │   ├── Server 2: users 10K-20K
        │   └── ...
        └── Redis Pub/Sub (routing layer)
            ├── Document channels: doc:{doc_id}
            └── Presence channels: presence:{doc_id}
        ```

        ---

        ## WebSocket Server Implementation

        ```python
        import asyncio
        import websockets
        import redis.asyncio as redis
        import json
        from typing import Dict, Set

        class WebSocketServer:
            """
            WebSocket server for real-time collaboration
            Handles 10K concurrent connections per instance
            """

            def __init__(self, server_id: str):
                self.server_id = server_id
                self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
                self.document_users: Dict[str, Set[str]] = {}  # doc_id -> set of user_ids
                self.redis = redis.Redis(host='redis-host')
                self.ot_server = OTServer()

            async def register(self, user_id: str, doc_id: str, websocket):
                """
                Register new WebSocket connection

                Args:
                    user_id: User connecting
                    doc_id: Document being edited
                    websocket: WebSocket connection
                """
                connection_id = f"{user_id}:{doc_id}"
                self.connections[connection_id] = websocket

                # Track users per document
                if doc_id not in self.document_users:
                    self.document_users[doc_id] = set()
                self.document_users[doc_id].add(user_id)

                logger.info(f"User {user_id} connected to doc {doc_id}. Total connections: {len(self.connections)}")

                # Subscribe to document channel
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(f"doc:{doc_id}")

                # Send current collaborators
                await self._send_collaborators(websocket, doc_id)

                # Broadcast user joined
                await self._broadcast_presence(doc_id, user_id, "joined")

                try:
                    # Listen for messages from client
                    async for message in websocket:
                        await self._handle_message(user_id, doc_id, message)

                    # Also listen for Redis pub/sub messages
                    asyncio.create_task(self._listen_redis(pubsub, websocket))

                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"User {user_id} disconnected from doc {doc_id}")
                finally:
                    # Cleanup
                    del self.connections[connection_id]
                    self.document_users[doc_id].discard(user_id)
                    await pubsub.unsubscribe(f"doc:{doc_id}")
                    await self._broadcast_presence(doc_id, user_id, "left")

            async def _handle_message(self, user_id: str, doc_id: str, message: str):
                """
                Handle incoming message from client

                Message types:
                - operation: Text edit operation
                - cursor: Cursor position update
                - selection: Text selection update
                """
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'operation':
                    await self._handle_operation(user_id, doc_id, data)
                elif msg_type == 'cursor':
                    await self._handle_cursor(user_id, doc_id, data)
                elif msg_type == 'selection':
                    await self._handle_selection(user_id, doc_id, data)

            async def _handle_operation(self, user_id: str, doc_id: str, data: dict):
                """
                Handle text edit operation

                Flow:
                1. Parse operation
                2. Transform against concurrent operations (OT)
                3. Apply to document
                4. Broadcast to collaborators
                5. Acknowledge to sender
                """
                operation = Operation(
                    type=OpType(data['operation']['type']),
                    position=data['operation']['position'],
                    content=data['operation']['content'],
                    attributes=data['operation'].get('attributes')
                )
                base_version = data['base_version']

                # Transform and apply operation
                transformed_op, new_version = await self.ot_server.handle_operation(
                    doc_id, user_id, operation, base_version
                )

                # Send acknowledgment to sender
                connection_id = f"{user_id}:{doc_id}"
                if connection_id in self.connections:
                    await self.connections[connection_id].send(json.dumps({
                        'type': 'operation_ack',
                        'op_id': data['op_id'],
                        'new_version': new_version,
                        'status': 'applied'
                    }))

                # Broadcast to other collaborators (handled by Redis pub/sub)

            async def _handle_cursor(self, user_id: str, doc_id: str, data: dict):
                """
                Handle cursor position update

                Broadcast to all collaborators
                """
                await self.redis.publish(f"doc:{doc_id}", json.dumps({
                    'type': 'cursor',
                    'user_id': user_id,
                    'cursor': data['cursor']
                }))

            async def _handle_selection(self, user_id: str, doc_id: str, data: dict):
                """Handle text selection update"""
                await self.redis.publish(f"doc:{doc_id}", json.dumps({
                    'type': 'selection',
                    'user_id': user_id,
                    'selection': data['selection']
                }))

            async def _listen_redis(self, pubsub, websocket):
                """
                Listen for Redis pub/sub messages and forward to WebSocket

                This allows broadcasting across multiple WebSocket servers
                """
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        # Forward to WebSocket client
                        await websocket.send(message['data'])

            async def _send_collaborators(self, websocket, doc_id: str):
                """Send list of current collaborators"""
                collaborators = list(self.document_users.get(doc_id, set()))
                await websocket.send(json.dumps({
                    'type': 'collaborators',
                    'users': collaborators
                }))

            async def _broadcast_presence(self, doc_id: str, user_id: str, action: str):
                """Broadcast user joined/left"""
                await self.redis.publish(f"doc:{doc_id}", json.dumps({
                    'type': 'presence',
                    'user_id': user_id,
                    'action': action
                }))
        ```

        ---

        ## Client-Side Optimization

        **Optimistic updates:** Apply edits locally before server acknowledgment.

        ```javascript
        class DocumentEditor {
          constructor(docId, userId) {
            this.docId = docId;
            this.userId = userId;
            this.version = 0;
            this.pendingOps = [];  // Operations waiting for ACK
            this.websocket = null;
          }

          async connect() {
            // Connect to WebSocket
            this.websocket = new WebSocket(`wss://docs.com/ws?doc=${this.docId}`);

            this.websocket.onmessage = (event) => {
              const data = JSON.parse(event.data);
              this.handleMessage(data);
            };
          }

          insert(position, text) {
            // 1. Apply locally (optimistic update)
            this.applyLocalOperation({
              type: 'insert',
              position: position,
              content: text
            });

            // 2. Send to server
            const operation = {
              type: 'operation',
              op_id: this.generateOpId(),
              document_id: this.docId,
              base_version: this.version,
              operation: {
                type: 'insert',
                position: position,
                content: text
              }
            };

            this.websocket.send(JSON.stringify(operation));
            this.pendingOps.push(operation);
          }

          handleMessage(data) {
            switch (data.type) {
              case 'operation_ack':
                // Server acknowledged our operation
                this.version = data.new_version;
                this.pendingOps = this.pendingOps.filter(op => op.op_id !== data.op_id);
                break;

              case 'remote_operation':
                // Another user's operation
                this.applyRemoteOperation(data.operation);
                this.version = data.version;
                break;

              case 'cursor':
                // Show other user's cursor
                this.showRemoteCursor(data.user_id, data.cursor);
                break;
            }
          }

          applyLocalOperation(operation) {
            // Apply to local document immediately
            const editor = document.getElementById('editor');
            if (operation.type === 'insert') {
              const content = editor.value;
              editor.value = content.slice(0, operation.position) +
                            operation.content +
                            content.slice(operation.position);
            }
          }

          applyRemoteOperation(operation) {
            // Transform against pending operations
            let transformed = operation;
            for (const pending of this.pendingOps) {
              transformed = this.transform(transformed, pending.operation);
            }

            // Apply transformed operation
            this.applyLocalOperation(transformed);
          }
        }
        ```

        ---

        ## Bandwidth Optimization

        **Problem:** Broadcasting every keystroke to 100 collaborators = high bandwidth.

        **Solutions:**

        1. **Operation batching:** Send operations every 50ms instead of every keystroke
        2. **Compression:** Use delta compression (only send changes)
        3. **Debouncing:** For cursor updates, send max 10 updates/sec

        ```python
        class OperationBatcher:
            """
            Batch multiple operations to reduce network traffic

            Example: User types "Hello" (5 keystrokes)
            Without batching: 5 network requests
            With batching: 1 network request with 5 operations
            """

            BATCH_INTERVAL = 0.05  # 50ms

            def __init__(self, websocket):
                self.websocket = websocket
                self.pending_ops = []
                self.batch_task = None

            def add_operation(self, operation: Operation):
                """Add operation to batch"""
                self.pending_ops.append(operation)

                # Start batch timer if not running
                if not self.batch_task:
                    self.batch_task = asyncio.create_task(self._flush_batch())

            async def _flush_batch(self):
                """Flush batch after timeout"""
                await asyncio.sleep(self.BATCH_INTERVAL)

                if self.pending_ops:
                    # Compose operations (merge consecutive operations)
                    composed = OT().compose(self.pending_ops)

                    # Send batch
                    await self.websocket.send(json.dumps({
                        'type': 'operation_batch',
                        'operations': [
                            {
                                'type': op.type.value,
                                'position': op.position,
                                'content': op.content
                            }
                            for op in composed
                        ]
                    }))

                    self.pending_ops = []

                self.batch_task = None
        ```

    === "📜 Version History"

        ## The Challenge

        **Problem:** Store infinite versions of 1B documents efficiently.

        **Naive approach:** Store full document for each version.

        ```
        1B documents × 100 versions × 50 KB = 5 PB (too expensive!)
        ```

        **Requirements:**

        - **Fast restore:** Load any version in < 1s
        - **Storage efficient:** Delta compression
        - **Long retention:** 30 days (full), 1 year (daily snapshots)
        - **Audit trail:** Who changed what, when

        ---

        ## Version Storage Strategy

        **Three-tier storage:**

        1. **Hot storage (Redis):** Last 10 versions (instant access)
        2. **Warm storage (Cassandra):** Last 1000 versions (delta-compressed)
        3. **Cold storage (S3):** Daily snapshots (full document)

        ---

        ## Snapshot + Delta Implementation

        ```python
        import json
        import gzip
        from typing import List, Optional
        from datetime import datetime, timedelta

        class VersionManager:
            """
            Manage document versions with snapshot + delta strategy

            Strategy:
            - Checkpoint every 1000 operations (snapshot)
            - Store deltas between checkpoints
            - Daily snapshots for long-term history
            """

            CHECKPOINT_INTERVAL = 1000  # Snapshot every 1000 operations

            def __init__(self, redis_client, cassandra_client, s3_client):
                self.redis = redis_client
                self.cassandra = cassandra_client
                self.s3 = s3_client

            async def save_version(self, doc_id: str, version: int,
                                  document: str, operation: Operation):
                """
                Save document version

                Args:
                    doc_id: Document ID
                    version: Version number
                    document: Full document content
                    operation: Operation that created this version
                """
                # 1. Save to Redis (hot storage - last 10 versions)
                await self._save_to_redis(doc_id, version, document, operation)

                # 2. Check if checkpoint needed
                if version % self.CHECKPOINT_INTERVAL == 0:
                    # Create snapshot
                    await self._create_checkpoint(doc_id, version, document)

                # 3. Save delta to Cassandra (warm storage)
                await self._save_delta(doc_id, version, operation)

                # 4. Daily snapshot to S3 (cold storage)
                await self._check_daily_snapshot(doc_id, document)

            async def _save_to_redis(self, doc_id: str, version: int,
                                    document: str, operation: Operation):
                """Save to Redis (last 10 versions)"""
                version_key = f"doc:{doc_id}:versions"

                version_data = {
                    'version': version,
                    'document': document,
                    'operation': {
                        'type': operation.type.value,
                        'position': operation.position,
                        'content': operation.content
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }

                # Add to Redis list
                await self.redis.lpush(version_key, json.dumps(version_data))

                # Keep only last 10 versions
                await self.redis.ltrim(version_key, 0, 9)

            async def _create_checkpoint(self, doc_id: str, version: int, document: str):
                """
                Create checkpoint (full snapshot)

                Checkpoints allow fast restore without replaying all operations
                """
                checkpoint_data = {
                    'document_id': doc_id,
                    'version': version,
                    'content': document,
                    'created_at': datetime.utcnow().isoformat()
                }

                # Compress checkpoint
                compressed = gzip.compress(json.dumps(checkpoint_data).encode())

                # Save to Cassandra
                await self.cassandra.execute(
                    """INSERT INTO checkpoints (document_id, version, content, created_at)
                       VALUES (%s, %s, %s, %s)""",
                    (doc_id, version, compressed, datetime.utcnow())
                )

                logger.info(f"Created checkpoint for doc {doc_id} at version {version}")

            async def _save_delta(self, doc_id: str, version: int, operation: Operation):
                """
                Save delta (operation only)

                Deltas are much smaller than full document
                """
                delta_data = {
                    'type': operation.type.value,
                    'position': operation.position,
                    'content': operation.content,
                    'attributes': operation.attributes
                }

                await self.cassandra.execute(
                    """INSERT INTO operations (document_id, version, operation, timestamp)
                       VALUES (%s, %s, %s, %s)""",
                    (doc_id, version, json.dumps(delta_data), datetime.utcnow())
                )

            async def _check_daily_snapshot(self, doc_id: str, document: str):
                """
                Create daily snapshot to S3

                Daily snapshots for long-term version history
                """
                today = datetime.utcnow().date()
                snapshot_key = f"doc:{doc_id}:daily:{today}"

                # Check if today's snapshot exists
                exists = await self.redis.exists(snapshot_key)
                if not exists:
                    # Create snapshot
                    snapshot_data = {
                        'document_id': doc_id,
                        'date': today.isoformat(),
                        'content': document
                    }

                    # Compress and upload to S3
                    compressed = gzip.compress(json.dumps(snapshot_data).encode())

                    s3_key = f"snapshots/{doc_id}/{today.isoformat()}.json.gz"
                    await self.s3.put_object(
                        Bucket='docs-versions',
                        Key=s3_key,
                        Body=compressed
                    )

                    # Mark as created (TTL: 25 hours)
                    await self.redis.setex(snapshot_key, 90000, "1")

                    logger.info(f"Created daily snapshot for doc {doc_id}")

            async def restore_version(self, doc_id: str, target_version: int) -> str:
                """
                Restore document to specific version

                Args:
                    doc_id: Document ID
                    target_version: Version to restore

                Returns:
                    Document content at target version
                """
                # 1. Check Redis (hot storage)
                redis_doc = await self._restore_from_redis(doc_id, target_version)
                if redis_doc:
                    return redis_doc

                # 2. Find nearest checkpoint
                checkpoint_version = (target_version // self.CHECKPOINT_INTERVAL) * self.CHECKPOINT_INTERVAL

                # Load checkpoint
                checkpoint = await self._load_checkpoint(doc_id, checkpoint_version)
                if not checkpoint:
                    # Fallback: Load from S3 daily snapshot
                    checkpoint = await self._load_daily_snapshot(doc_id, target_version)

                document = checkpoint['content']

                # 3. Replay operations from checkpoint to target version
                operations = await self._load_operations(doc_id, checkpoint_version + 1, target_version)

                ot = OperationalTransform()
                for op_data in operations:
                    operation = Operation(
                        type=OpType(op_data['type']),
                        position=op_data['position'],
                        content=op_data['content'],
                        attributes=op_data.get('attributes')
                    )
                    document = ot.apply_operation(document, operation)

                return document

            async def _restore_from_redis(self, doc_id: str, version: int) -> Optional[str]:
                """Try to restore from Redis (last 10 versions)"""
                versions_json = await self.redis.lrange(f"doc:{doc_id}:versions", 0, -1)

                for version_json in versions_json:
                    version_data = json.loads(version_json)
                    if version_data['version'] == version:
                        return version_data['document']

                return None

            async def _load_checkpoint(self, doc_id: str, version: int) -> Optional[dict]:
                """Load checkpoint from Cassandra"""
                result = await self.cassandra.execute(
                    """SELECT content FROM checkpoints
                       WHERE document_id = %s AND version = %s""",
                    (doc_id, version)
                )

                if result:
                    compressed = result[0]['content']
                    decompressed = gzip.decompress(compressed)
                    return json.loads(decompressed)

                return None

            async def _load_operations(self, doc_id: str, from_version: int,
                                      to_version: int) -> List[dict]:
                """Load operations from Cassandra"""
                results = await self.cassandra.execute(
                    """SELECT operation FROM operations
                       WHERE document_id = %s AND version >= %s AND version <= %s
                       ORDER BY version ASC""",
                    (doc_id, from_version, to_version)
                )

                return [json.loads(row['operation']) for row in results]
        ```

        ---

        ## Version History UI

        **Timeline view:**

        ```
        Today
        ├── 14:30 - John Doe added bold formatting (version 1524)
        ├── 14:25 - Jane Smith inserted "Hello World" (version 1523)
        ├── 14:20 - John Doe created document (version 1)

        Yesterday
        ├── 16:45 - Jane Smith added image (version 1200)
        └── 10:00 - John Doe edited paragraph 2 (version 1000)

        [Show daily snapshots] ▼
        ```

        **Version comparison:**

        ```
        Version 1523                  Version 1524
        ┌─────────────────────┐      ┌─────────────────────┐
        │ Hello World         │      │ Hello World         │
        │                     │  →   │ **Bold text added** │
        │ Lorem ipsum...      │      │ Lorem ipsum...      │
        └─────────────────────┘      └─────────────────────┘

        [Restore this version] [Show details]
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Google Docs from 1M to 2B users.

    **Scaling challenges at 2B users:**

    - **Write throughput:** 7M edit ops/sec
    - **Read throughput:** 3.3M view requests/sec
    - **WebSocket connections:** 100M concurrent connections
    - **Storage:** 2 PB of document data

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **WebSocket servers** | ✅ Yes | 10,000 servers (10K connections each), Redis pub/sub routing |
    | **OT Engine** | ✅ Yes | Stateless OT servers (100 instances), Redis for OT state |
    | **Cassandra writes** | ✅ Yes | 500 nodes, SSD storage, append-only operations |
    | **MongoDB reads** | ✅ Yes | Sharding by doc_id, read replicas, caching (Redis) |
    | **Redis** | 🟡 Approaching | 200 Redis instances, Redis Cluster for sharding |

    ---

    ## Sharding Strategy

    ### Document Sharding (MongoDB)

    **Shard key:** `document_id` (hash-based)

    ```
    1B documents
    ├── Shard 0: documents 0-20M (20M docs)
    ├── Shard 1: documents 20M-40M (20M docs)
    ├── ...
    └── Shard 49: documents 980M-1B (20M docs)

    Total: 50 shards × 3 replicas = 150 MongoDB instances
    ```

    **Benefits:**

    - **Even distribution:** Hash-based sharding
    - **Scalability:** Add shards as documents grow
    - **Isolation:** Hot documents don't affect others

    ---

    ### Operations Log Sharding (Cassandra)

    **Partition key:** `document_id`
    **Clustering key:** `version` (ascending)

    ```
    operations table
    ├── Partition: doc_abc123
    │   ├── version 1
    │   ├── version 2
    │   └── ... (ordered by version)
    └── Partition: doc_xyz789
        └── ...
    ```

    **Benefits:**

    - **Write-optimized:** Append-only, 7M ops/sec
    - **Fast range queries:** Get operations [v1...v100] in one query
    - **Horizontal scaling:** Add nodes for more throughput

    ---

    ## Caching Strategy

    ### Multi-tier Caching

    **L1 Cache (Browser):**
    - Store document locally (IndexedDB)
    - Cache for offline editing
    - 50 KB per document

    **L2 Cache (Redis):**
    - Hot documents (accessed in last 5 minutes)
    - TTL: 300 seconds
    - Hit rate: 80%

    **L3 Cache (CDN):**
    - Static assets (JavaScript, CSS)
    - Document snapshots (read-only)

    ---

    ### Redis Caching Implementation

    ```python
    class DocumentCache:
        """
        Multi-tier caching for documents

        Cache hierarchy:
        1. Redis (hot documents)
        2. MongoDB (warm documents)
        3. S3 (cold snapshots)
        """

        CACHE_TTL = 300  # 5 minutes

        def __init__(self, redis_client, mongodb_client, s3_client):
            self.redis = redis_client
            self.mongodb = mongodb_client
            self.s3 = s3_client

        async def get_document(self, doc_id: str) -> Optional[dict]:
            """
            Get document with caching

            Args:
                doc_id: Document ID

            Returns:
                Document data or None
            """
            # L1: Check Redis
            cache_key = f"doc:{doc_id}"
            cached = await self.redis.get(cache_key)

            if cached:
                logger.info(f"Cache HIT for doc {doc_id}")
                return json.loads(cached)

            logger.info(f"Cache MISS for doc {doc_id}")

            # L2: Query MongoDB
            doc = await self.mongodb.find_one({'_id': doc_id})

            if doc:
                # Cache for 5 minutes
                await self.redis.setex(
                    cache_key,
                    self.CACHE_TTL,
                    json.dumps(doc)
                )
                return doc

            return None

        async def invalidate_cache(self, doc_id: str):
            """Invalidate cache when document is edited"""
            await self.redis.delete(f"doc:{doc_id}")
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 2B users:**

    | Component | Cost |
    |-----------|------|
    | **WebSocket servers** | $216,000 (10,000 × m5.large) |
    | **API servers** | $43,200 (300 × m5.xlarge) |
    | **MongoDB cluster** | $86,400 (150 nodes) |
    | **Cassandra cluster** | $216,000 (500 nodes) |
    | **Redis cache** | $86,400 (200 nodes) |
    | **S3 storage** | $46,000 (2 PB) |
    | **CDN** | $85,000 (1,000 TB egress) |
    | **Kafka cluster** | $21,600 (50 brokers) |
    | **Total** | **$800,600/month** |

    **Cost optimizations:**

    1. **Tiered storage:** Hot (SSD), warm (HDD), cold (S3 Glacier)
    2. **Compression:** Gzip for snapshots (50% reduction)
    3. **CDN caching:** Cache read-only documents (80% hit rate)
    4. **Spot instances:** Use for batch processing (snapshot generation)

    ---

    ## Performance Optimizations

    ### 1. Operation Compression

    **Problem:** Sending every keystroke = high bandwidth.

    **Solution:** Compress consecutive operations.

    ```python
    # Before compression:
    operations = [
        Operation(INSERT, 0, "H"),
        Operation(INSERT, 1, "e"),
        Operation(INSERT, 2, "l"),
        Operation(INSERT, 3, "l"),
        Operation(INSERT, 4, "o"),
    ]  # 5 operations

    # After compression:
    operations = [
        Operation(INSERT, 0, "Hello")
    ]  # 1 operation (5x reduction)
    ```

    ---

    ### 2. Cursor Throttling

    **Problem:** Sending cursor updates every 10ms = 100 updates/sec.

    **Solution:** Throttle to 10 updates/sec (100ms interval).

    ```javascript
    class CursorThrottler {
      constructor(sendFunction, interval = 100) {
        this.send = sendFunction;
        this.interval = interval;
        this.lastSent = 0;
        this.pending = null;
      }

      updateCursor(position) {
        const now = Date.now();

        if (now - this.lastSent >= this.interval) {
          // Send immediately
          this.send({ type: 'cursor', position });
          this.lastSent = now;
        } else {
          // Throttle: save for later
          this.pending = position;

          if (!this.timer) {
            this.timer = setTimeout(() => {
              if (this.pending !== null) {
                this.send({ type: 'cursor', position: this.pending });
                this.lastSent = Date.now();
                this.pending = null;
              }
              this.timer = null;
            }, this.interval - (now - this.lastSent));
          }
        }
      }
    }
    ```

    ---

    ### 3. Presence Optimization

    **Problem:** Broadcasting user status to 100 collaborators every second.

    **Solution:** Use heartbeat + cache.

    ```python
    class PresenceManager:
        """
        Manage user presence with heartbeat

        Users send heartbeat every 30s
        If no heartbeat for 60s, mark as offline
        """

        HEARTBEAT_INTERVAL = 30  # seconds
        OFFLINE_THRESHOLD = 60   # seconds

        async def heartbeat(self, doc_id: str, user_id: str):
            """Record user heartbeat"""
            await self.redis.setex(
                f"presence:{doc_id}:{user_id}",
                self.OFFLINE_THRESHOLD,
                json.dumps({
                    'user_id': user_id,
                    'last_seen': time.time(),
                    'status': 'online'
                })
            )

        async def get_online_users(self, doc_id: str) -> List[str]:
            """Get list of online users"""
            pattern = f"presence:{doc_id}:*"
            keys = await self.redis.keys(pattern)

            users = []
            for key in keys:
                user_data = await self.redis.get(key)
                if user_data:
                    users.append(json.loads(user_data)['user_id'])

            return users
    ```

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Edit Sync Latency (P95)** | < 100ms | > 200ms |
    | **Document Load Time (P95)** | < 500ms | > 1s |
    | **WebSocket Connection Success** | > 99.9% | < 99% |
    | **OT Conflict Rate** | < 1% | > 5% |
    | **Cache Hit Rate** | > 80% | < 70% |
    | **Operation Throughput** | 7M ops/sec | < 5M ops/sec |

    **Alerting rules:**

    ```yaml
    alerts:
      - name: HighEditLatency
        condition: p95(edit_sync_latency) > 200ms for 5 minutes
        severity: critical
        action: Page on-call engineer

      - name: WebSocketConnectionFailure
        condition: websocket_connection_success_rate < 99% for 2 minutes
        severity: critical
        action: Page on-call engineer

      - name: LowCacheHitRate
        condition: cache_hit_rate < 70% for 10 minutes
        severity: warning
        action: Notify team
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Operational Transformation:** Deterministic conflict resolution for rich text
    2. **WebSocket architecture:** 100M concurrent connections via Redis pub/sub
    3. **Optimistic updates:** Apply edits locally, sync in background
    4. **Snapshot + delta:** Efficient version storage (checkpoints every 1000 ops)
    5. **Multi-tier caching:** Redis (hot), MongoDB (warm), S3 (cold)
    6. **Eventual consistency:** All clients converge to same state

    ---

    ## Interview Tips

    ✅ **Emphasize OT vs CRDT trade-offs** - OT better for rich text, CRDT for offline

    ✅ **Discuss real-time sync architecture** - WebSocket + Redis pub/sub + optimistic updates

    ✅ **Explain conflict resolution** - OT transformation functions in detail

    ✅ **Cover version history strategy** - Snapshot + delta compression

    ✅ **Address offline editing** - Local storage + conflict resolution on reconnect

    ✅ **Scale WebSocket connections** - 10K per server, Redis routing

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle 100 concurrent editors?"** | OT engine transforms all operations, Redis for state, WebSocket broadcast |
    | **"How to resolve conflicts?"** | Operational Transformation with transformation functions, deterministic ordering |
    | **"How to support offline editing?"** | Local storage (IndexedDB), queue operations, transform against server state on reconnect |
    | **"How to show other users' cursors?"** | WebSocket broadcast cursor positions (throttled to 10 updates/sec), render in UI |
    | **"How to implement version history?"** | Snapshot every 1000 operations, store deltas, replay operations for restore |
    | **"OT vs CRDT - when to use each?"** | OT: rich text, lower bandwidth. CRDT: offline-first, peer-to-peer |

    ---

    ## Edge Cases to Consider

    1. **Network partition:** Users edit offline, need conflict resolution on reconnect
    2. **Large documents:** Split into sections, lazy load (only load visible sections)
    3. **Slow clients:** Buffer operations server-side, apply in batches
    4. **Malicious clients:** Rate limiting (100 ops/sec per user), validation
    5. **Document forking:** User restores old version, creates conflict
    6. **Race conditions:** Use vector clocks or Lamport timestamps for ordering

    ---

    ## Advanced Topics (if time permits)

    ### 1. Rich Text Formatting

    **Problem:** How to handle formatting (bold, italic, etc.) with OT?

    **Solution:** Use Quill Delta format (operation + attributes).

    ```javascript
    {
      ops: [
        { insert: "Hello " },
        { insert: "World", attributes: { bold: true } },
        { insert: "\n" }
      ]
    }
    ```

    ---

    ### 2. Collaborative Cursors

    **Implementation:**

    ```javascript
    class CursorManager {
      constructor() {
        this.cursors = {};  // user_id -> cursor position
      }

      updateCursor(userId, position, color) {
        this.cursors[userId] = { position, color };
        this.renderCursor(userId);
      }

      renderCursor(userId) {
        const cursor = this.cursors[userId];
        const cursorEl = document.createElement('div');
        cursorEl.className = 'remote-cursor';
        cursorEl.style.backgroundColor = cursor.color;
        cursorEl.style.left = `${this.getPositionX(cursor.position)}px`;
        cursorEl.style.top = `${this.getPositionY(cursor.position)}px`;
        document.body.appendChild(cursorEl);
      }
    }
    ```

    ---

    ### 3. Comments & Suggestions

    **Data model:**

    ```javascript
    {
      comment_id: "comment_123",
      document_id: "doc_abc",
      user_id: "user_456",
      content: "Great point!",
      position: {
        start: 42,
        end: 57
      },
      created_at: "2026-02-02T14:30:00Z",
      resolved: false,
      replies: [
        {
          user_id: "user_789",
          content: "Thanks!",
          created_at: "2026-02-02T14:35:00Z"
        }
      ]
    }
    ```

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Google, Microsoft (Office 365), Notion, Dropbox Paper, Confluence

---

*Master this problem and you'll be ready for: Google Docs, Microsoft Office 365, Notion, Dropbox Paper, Confluence, Figma*
