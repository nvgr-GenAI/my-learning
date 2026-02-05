# Design Slack

Design a team collaboration platform with real-time messaging, channels, direct messages, file sharing, search, and integrations. Support millions of teams with real-time message delivery and message history.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M DAU, 20M concurrent connections, 10B messages/day |
| **Key Challenges** | Real-time delivery, message ordering, presence management, search at scale |
| **Core Concepts** | WebSocket, message queue, eventual consistency, full-text search, presence |
| **Companies** | Slack, Microsoft Teams, Discord, Zoom, Meta |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Channels** | Public/private group conversations | P0 (Must have) |
    | **Direct Messages** | 1-on-1 and group DMs | P0 (Must have) |
    | **Real-time Messaging** | Instant message delivery | P0 (Must have) |
    | **File Sharing** | Upload and share files in conversations | P0 (Must have) |
    | **Search** | Search messages, files, channels | P0 (Must have) |
    | **User Presence** | Online/away/offline status | P1 (Should have) |
    | **Threads** | Reply threads within messages | P1 (Should have) |
    | **Reactions** | Emoji reactions to messages | P1 (Should have) |
    | **Notifications** | Push and in-app notifications | P1 (Should have) |
    | **Integrations** | Third-party app integrations | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Video/audio calling
    - Screen sharing
    - Advanced admin controls
    - Custom emoji creation
    - Message translation
    - Slack Connect (cross-workspace)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Teams depend on it for work |
    | **Message Latency** | < 100ms p95 | Real-time feel is critical |
    | **Message Ordering** | Strict per channel | Messages must appear in order |
    | **Consistency** | Eventual consistency | All users eventually see same messages |
    | **Scalability** | 20M concurrent users | Handle enterprise teams |
    | **Search Latency** | < 200ms | Fast message search |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 10 million
    Concurrent connections: 20 million (business hours worldwide)
    Average messages per user per day: 50 messages
    Total messages per day: 10M × 50 = 500M messages/day

    QPS calculations:
    - Message send: 500M / 86,400 = 5,800 messages/sec
    - Peak: 3x average = 17,400 messages/sec
    - Message reads: 10x writes = 58,000 reads/sec
    - Presence updates: 20M users × 1/min = 333K updates/sec
    - File uploads: 10M files/day = 116 uploads/sec
    ```

    ### Storage Estimates

    ```
    Messages:
    - Message content: 500 bytes (average)
    - Metadata: 200 bytes (timestamp, user, channel)
    - Total per message: 700 bytes

    Daily storage:
    - 500M messages × 700 bytes = 350 GB/day
    - Monthly: 350 GB × 30 = 10.5 TB/month
    - Yearly: 10.5 TB × 12 = 126 TB/year

    Files:
    - 10M files/day × 2 MB average = 20 TB/day
    - Monthly: 600 TB/month
    - Yearly: 7.2 PB/year

    User data:
    - 10M users × 10 KB = 100 GB

    Workspace metadata:
    - 1M workspaces × 100 KB = 100 GB

    Total (1 year): 126 TB (messages) + 7.2 PB (files) + 200 GB (user data) ≈ 7.3 PB
    ```

    ### Bandwidth Estimates

    ```
    Message ingress:
    - 5,800 messages/sec × 700 bytes = 4 MB/sec ≈ 32 Mbps

    Message egress (fan-out to recipients):
    - Average 20 users per channel
    - 5,800 messages/sec × 20 users × 700 bytes = 80 MB/sec ≈ 640 Mbps

    File uploads:
    - 116 files/sec × 2 MB = 232 MB/sec ≈ 1.86 Gbps

    File downloads:
    - 10x uploads = 2,320 MB/sec ≈ 18.6 Gbps

    Total ingress: ~2 Gbps
    Total egress: ~20 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Active connections (WebSocket):
    - 20M connections × 10 KB state = 200 GB

    Recent messages cache:
    - Last 1M messages × 700 bytes = 700 MB

    User presence:
    - 10M users × 100 bytes = 1 GB

    Channel metadata:
    - 10M channels × 1 KB = 10 GB

    Total cache: 200 GB + 700 MB + 1 GB + 10 GB ≈ 212 GB
    ```

    ---

    ## Key Assumptions

    1. Average channel has 20 members
    2. Users are in 10 channels on average
    3. 60% mobile, 40% desktop usage
    4. Message retention: unlimited (competitors offer limited free tier)
    5. Peak hours: overlap of global business hours (8am-12pm UTC)
    6. File size limit: 1 GB per file

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Real-time first** - WebSocket connections for instant delivery
    2. **Message ordering** - Lamport timestamps or sequence numbers per channel
    3. **Availability over consistency** - Accept eventual consistency
    4. **Horizontal scaling** - Stateless message processors
    5. **Fast search** - Elasticsearch for full-text search

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile Apps]
            Desktop[Desktop Apps]
            Web[Web Browser]
        end

        subgraph "Connection Layer"
            WS1[WebSocket Server 1]
            WS2[WebSocket Server 2]
            WS3[WebSocket Server N]
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Message_API[Message Service]
            Channel_API[Channel Service]
            User_API[User Service]
            File_API[File Service]
            Search_API[Search Service]
            Presence_API[Presence Service]
        end

        subgraph "Message Processing"
            MessageQueue[Kafka<br/>Message Queue]
            Fanout[Fanout Worker<br/>Deliver to recipients]
            SearchIndexer[Search Indexer<br/>Elasticsearch]
        end

        subgraph "Caching Layer"
            Redis_Session[Redis<br/>WebSocket Sessions]
            Redis_Presence[Redis<br/>User Presence]
            Redis_Message[Redis<br/>Recent Messages]
        end

        subgraph "Storage Layer"
            MessageDB[(Message Store<br/>Cassandra)]
            ChannelDB[(Channel/User DB<br/>PostgreSQL)]
            FileStore[(Object Storage<br/>S3)]
            SearchIndex[(Elasticsearch<br/>Full-text Search)]
        end

        Mobile --> LB
        Desktop --> LB
        Web --> LB

        LB --> WS1
        LB --> WS2
        LB --> WS3

        WS1 --> Message_API
        WS2 --> Message_API
        WS3 --> Message_API

        Message_API --> MessageQueue
        Message_API --> MessageDB
        Message_API --> Redis_Message

        MessageQueue --> Fanout
        MessageQueue --> SearchIndexer

        Fanout --> Redis_Session
        Fanout --> WS1
        Fanout --> WS2
        Fanout --> WS3

        SearchIndexer --> SearchIndex

        Channel_API --> ChannelDB
        User_API --> ChannelDB
        File_API --> FileStore
        Search_API --> SearchIndex
        Presence_API --> Redis_Presence

        style WS1 fill:#e1f5ff
        style MessageQueue fill:#fff4e1
        style Redis_Message fill:#ffe1e1
        style MessageDB fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Technology | Why This? | Alternative |
    |-----------|-----------|-----------|-------------|
    | **WebSocket** | Custom WS server | Real-time bidirectional, low latency | HTTP polling (wasteful), SSE (one-way only) |
    | **Message Queue** | Kafka | High throughput, replay capability, reliable | RabbitMQ (lower throughput), SQS (higher latency) |
    | **Message Store** | Cassandra | High write throughput, time-series optimized | MySQL (can't handle writes), MongoDB (scaling issues) |
    | **Channel/User DB** | PostgreSQL | ACID transactions, complex queries | NoSQL (harder to query), custom (complex) |
    | **File Storage** | S3 | Scalable, durable, CDN integration | Database (expensive), custom (complex) |
    | **Search** | Elasticsearch | Full-text search, fast, scalable | Database LIKE (too slow), custom (complex) |
    | **Presence** | Redis | Fast updates, TTL support, pub/sub | Database (too slow), custom (complex) |
    | **Session Store** | Redis | Fast lookups, routing WebSocket connections | Database (too slow), in-memory (not distributed) |

    ---

    ## API Design

    ### 1. Send Message

    **Request (via WebSocket):**
    ```json
    {
      "type": "message",
      "channel_id": "C123456",
      "text": "Hello team!",
      "thread_ts": null,
      "attachments": ["file_id_123"],
      "client_msg_id": "uuid-client-generated"
    }
    ```

    **Response:**
    ```json
    {
      "type": "message_ack",
      "ok": true,
      "message": {
        "message_id": "msg_789",
        "channel_id": "C123456",
        "user_id": "U123",
        "text": "Hello team!",
        "timestamp": "2026-01-29T10:30:00.123Z",
        "client_msg_id": "uuid-client-generated"
      }
    }
    ```

    ---

    ### 2. Create Channel

    **Request (REST API):**
    ```http
    POST /api/v1/channels
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "name": "engineering",
      "is_private": false,
      "description": "Engineering team discussions",
      "member_ids": ["U123", "U456", "U789"]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "channel_id": "C123456",
      "name": "engineering",
      "created_by": "U123",
      "created_at": "2026-01-29T10:30:00Z",
      "member_count": 3
    }
    ```

    ---

    ### 3. Search Messages

    **Request:**
    ```http
    GET /api/v1/search?q=deployment&in=channel:C123456&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "messages": [
        {
          "message_id": "msg_123",
          "channel_id": "C123456",
          "user": {
            "user_id": "U456",
            "name": "Jane Doe",
            "avatar": "..."
          },
          "text": "The deployment is scheduled for tonight",
          "timestamp": "2026-01-28T15:30:00Z",
          "highlight": "The <em>deployment</em> is scheduled for tonight"
        }
      ],
      "total": 47,
      "has_more": true
    }
    ```

    ---

    ### 4. Update Presence

    **Request (via WebSocket):**
    ```json
    {
      "type": "presence",
      "status": "active"
    }
    ```

    **Broadcast to team:**
    ```json
    {
      "type": "presence_change",
      "user_id": "U123",
      "status": "active",
      "timestamp": "2026-01-29T10:30:00Z"
    }
    ```

    ---

    ## Database Schema

    ### Messages (Cassandra)

    ```sql
    CREATE TABLE messages (
        channel_id TEXT,
        message_id TIMEUUID,
        user_id TEXT,
        text TEXT,
        attachments LIST<TEXT>,
        thread_ts TIMESTAMP,
        created_at TIMESTAMP,
        PRIMARY KEY (channel_id, message_id)
    ) WITH CLUSTERING ORDER BY (message_id DESC);

    CREATE TABLE user_messages (
        user_id TEXT,
        message_id TIMEUUID,
        channel_id TEXT,
        created_at TIMESTAMP,
        PRIMARY KEY (user_id, message_id)
    ) WITH CLUSTERING ORDER BY (message_id DESC);
    ```

    ---

    ### Channels & Users (PostgreSQL)

    ```sql
    CREATE TABLE workspaces (
        workspace_id BIGINT PRIMARY KEY,
        name VARCHAR(100),
        domain VARCHAR(100) UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        workspace_id BIGINT REFERENCES workspaces(workspace_id),
        email VARCHAR(255) UNIQUE,
        name VARCHAR(100),
        avatar_url TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_workspace_id (workspace_id)
    );

    CREATE TABLE channels (
        channel_id BIGINT PRIMARY KEY,
        workspace_id BIGINT REFERENCES workspaces(workspace_id),
        name VARCHAR(80),
        is_private BOOLEAN DEFAULT FALSE,
        description TEXT,
        created_by BIGINT REFERENCES users(user_id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_workspace_id (workspace_id),
        INDEX idx_name (name)
    );

    CREATE TABLE channel_members (
        channel_id BIGINT REFERENCES channels(channel_id),
        user_id BIGINT REFERENCES users(user_id),
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (channel_id, user_id),
        INDEX idx_user_id (user_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Message Send Flow

    ```mermaid
    sequenceDiagram
        participant Client as Client (WS)
        participant WS as WebSocket Server
        participant MessageAPI as Message Service
        participant Kafka
        participant Cassandra
        participant Fanout as Fanout Worker
        participant Redis as Session Store

        Client->>WS: Send message (WebSocket)
        WS->>MessageAPI: Forward message
        MessageAPI->>MessageAPI: Validate, assign ID + timestamp
        MessageAPI->>Cassandra: Store message
        Cassandra-->>MessageAPI: Success
        MessageAPI->>Kafka: Publish message event
        MessageAPI-->>Client: ACK (message_id, timestamp)

        Kafka->>Fanout: Process message event
        Fanout->>Fanout: Get channel members
        Fanout->>Redis: Lookup online users' WS servers

        loop For each online member
            Fanout->>WS: Route to correct WS server
            WS->>Client: Deliver message (WebSocket)
        end

        Note over Fanout: Offline users get message on reconnect
    ```

    ### Real-time Presence Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant WS as WebSocket Server
        participant Presence as Presence Service
        participant Redis

        Client->>WS: Heartbeat (every 30s)
        WS->>Presence: Update presence
        Presence->>Redis: SET user:123:presence "active" EX 60
        Redis-->>Presence: OK

        Note over Presence: Broadcast to team members

        Presence->>Redis: GET team members
        Redis-->>Presence: [U456, U789, ...]

        loop For each team member
            Presence->>WS: Send presence_change event
            WS->>Client: User U123 is active
        end

        Note over Redis: After 60s without heartbeat<br/>key expires = user offline
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics to Cover:

    ### 1. WebSocket Connection Management
    - Connection routing (which WS server has which user)
    - Load balancing WebSocket connections
    - Heartbeat and reconnection logic
    - Graceful degradation (fallback to HTTP polling)

    ### 2. Message Ordering
    - Use Lamport timestamps or sequence numbers per channel
    - Handle clock skew across distributed servers
    - Ensure causal ordering (thread replies after parent)

    ### 3. Fanout Strategy
    - For small channels (<100 members): Fan-out on write
    - For large channels (>100 members): Fan-out on read with cache
    - Hybrid: Cache last N messages per channel

    ### 4. Search Implementation
    - Real-time indexing (< 1 second lag)
    - Relevance ranking (recency + keyword match)
    - Filters: channel, user, date range, has:link, has:file
    - Pagination and highlighting

    ### 5. Presence Management
    - Redis with TTL (expire after 60s without heartbeat)
    - Efficient broadcast (only to team members, not entire workspace)
    - Presence status: active, away, offline

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **WebSocket Servers** | ✅ Yes | Horizontal scaling, 10K connections per server = 2,000 servers |
    | **Message Queue** | 🟡 Maybe | Kafka partitioning by channel_id, 50 brokers |
    | **Cassandra** | ❌ No | Handles 50K writes/sec easily |
    | **Fanout Workers** | ✅ Yes | Async workers, batch deliveries, 100+ workers |

    ## Performance Optimization

    - **Connection pooling**: Reuse DB connections
    - **Message batching**: Send multiple messages in one WebSocket frame
    - **Lazy loading**: Load old messages on demand, not on join
    - **CDN for files**: Serve uploaded files via CDN
    - **Compression**: Gzip WebSocket messages

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **WebSocket for real-time** - Bidirectional, low latency
    2. **Kafka for reliability** - Message replay, guaranteed delivery
    3. **Cassandra for messages** - High write throughput, time-series
    4. **Elasticsearch for search** - Fast full-text search
    5. **Redis for presence** - Fast updates, TTL for offline detection
    6. **Hybrid fanout** - Balance write vs read fanout based on channel size

    ## Interview Tips

    ✅ **Emphasize real-time** - WebSocket architecture is critical
    ✅ **Message ordering** - Explain how to maintain order in distributed system
    ✅ **Fanout strategy** - Small vs large channels
    ✅ **Presence management** - Redis TTL approach
    ✅ **Search at scale** - Elasticsearch sharding and indexing

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle message ordering?"** | Use sequence numbers per channel, timestamp conflicts resolved by server |
    | **"What if WebSocket server crashes?"** | Client reconnects to another server (stateless), session store in Redis |
    | **"How to prevent message loss?"** | Write to Cassandra before ACK, Kafka for reliable delivery |
    | **"How to handle large channels?"** | Fan-out on read with caching, don't write to 10K user timelines |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Slack, Microsoft Teams, Discord, Zoom, Meta

---

*Master this problem and you'll understand real-time messaging systems, WebSocket architecture, and distributed message ordering.*
