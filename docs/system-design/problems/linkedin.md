# Design LinkedIn

Design a professional networking platform where users can create profiles, connect with professionals, post updates, search for jobs, and engage with content through likes, comments, and shares.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 900M users, 60M DAU, 3M job posts, 100M daily posts |
| **Key Challenges** | Social graph complexity, job matching, news feed generation, search at scale |
| **Core Concepts** | Graph database, feed ranking, job recommendation, real-time messaging, connections |
| **Companies** | LinkedIn, Meta, Google, Amazon, Microsoft |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **User Profile** | Create/edit professional profile with experience, skills | P0 (Must have) |
    | **Connections** | Send/accept connection requests, view network | P0 (Must have) |
    | **News Feed** | View posts from connections and followed companies | P0 (Must have) |
    | **Job Posting** | Companies post jobs, users apply | P0 (Must have) |
    | **Search** | Search people, jobs, companies, posts | P0 (Must have) |
    | **Messaging** | Direct messaging between connections | P1 (Should have) |
    | **Recommendations** | People you may know, jobs you may like | P1 (Should have) |
    | **Skills Endorsement** | Endorse connections for skills | P1 (Should have) |
    | **Groups** | Professional groups and discussions | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - LinkedIn Learning (courses)
    - Premium subscriptions
    - Sales Navigator
    - Recruiter tools
    - Analytics dashboard
    - Ads system

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Business networking must be reliable |
    | **Feed Latency** | < 500ms | Fast feed loading for engagement |
    | **Search Latency** | < 200ms | Quick search results critical |
    | **Scalability** | Billions of connections | Handle massive social graph |
    | **Consistency** | Eventual consistency | Brief delays acceptable |
    | **Real-time** | Messages delivered < 1s | Professional communication needs speed |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total users: 900 million
    Daily Active Users (DAU): 60 million (7%)
    Monthly Active Users (MAU): 300 million (33%)

    Daily actions:
    - Profile views: 60M × 10 = 600M views/day
    - Connection requests: 60M × 2 = 120M requests/day
    - Posts created: 60M × 1.67 = 100M posts/day
    - Job applications: 5M applications/day
    - Searches: 60M × 3 = 180M searches/day
    - Messages sent: 60M × 5 = 300M messages/day

    QPS calculations:
    - Profile views: 600M / 86,400 = 7,000 QPS
    - Feed generation: 600M / 86,400 = 7,000 QPS
    - Search: 180M / 86,400 = 2,100 QPS
    - Job matching: 5M / 86,400 = 58 QPS
    - Messaging: 300M / 86,400 = 3,500 QPS
    ```

    ### Storage Estimates

    ```
    User profiles:
    - Profile data: 10 KB per user
    - Profile photo: 200 KB per user
    - 900M users × 210 KB = 189 TB

    Connections (social graph):
    - Average 500 connections per user
    - 900M × 500 × 16 bytes = 7.2 TB

    Posts:
    - 100M posts/day × 2 KB = 200 GB/day
    - For 5 years: 200 GB × 365 × 5 = 365 TB

    Job posts:
    - 3M active jobs × 5 KB = 15 GB
    - Historical (10 years): 500 GB

    Messages:
    - 300M messages/day × 1 KB = 300 GB/day
    - For 1 year: 300 GB × 365 = 109.5 TB

    Total storage: 189 TB + 7.2 TB + 365 TB + 500 GB + 109.5 TB ≈ 671 TB
    ```

    ### Bandwidth Estimates

    ```
    Read bandwidth:
    - Profile views: 7,000 QPS × 210 KB = 1.47 GB/sec ≈ 12 Gbps
    - Feed: 7,000 QPS × 50 KB = 350 MB/sec ≈ 2.8 Gbps
    - Search: 2,100 QPS × 10 KB = 21 MB/sec ≈ 168 Mbps

    Total read: ~15 Gbps

    Write bandwidth:
    - Posts: 100M/day × 2 KB = 2.3 MB/sec ≈ 18 Mbps
    - Messages: 300M/day × 1 KB = 3.5 MB/sec ≈ 28 Mbps
    - Connection updates: 120M/day × 100 bytes = 139 KB/sec

    Total write: ~50 Mbps
    ```

    ### Memory Estimates (Caching)

    ```
    Hot user profiles (online users):
    - 5M concurrent users × 210 KB = 1 TB

    Feed cache (pre-generated timelines):
    - 10M most active users × 100 KB = 1 TB

    Connection graph cache:
    - 100M most active users × 50 KB = 5 TB

    Search index cache:
    - 100 GB

    Total cache: 1 TB + 1 TB + 5 TB + 100 GB ≈ 7.1 TB
    ```

    ---

    ## Key Assumptions

    1. Average 500 connections per user (following power law distribution)
    2. 60% mobile, 40% web/desktop usage
    3. Feed shows posts from 1st and 2nd degree connections
    4. Jobs expire after 30 days
    5. Most activity during business hours (9am-5pm)
    6. Read-heavy workload (100:1 read/write ratio)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Graph-first architecture** - Social connections are core to the platform
    2. **Feed personalization** - ML-ranked feed based on relevance
    3. **Search-centric** - Everything is searchable (people, jobs, posts, companies)
    4. **Real-time messaging** - Low-latency chat between connections
    5. **Job matching** - ML-based job recommendations

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Application]
            Mobile[Mobile Apps]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Load Balancer]
        end

        subgraph "Service Layer"
            Profile[Profile Service]
            Connection[Connection Service]
            Feed[Feed Service]
            Job[Job Service]
            Search[Search Service]
            Message[Messaging Service]
            Notification[Notification Service]
        end

        subgraph "Data Processing"
            FeedWorker[Feed Generator<br/>Kafka Consumer]
            JobMatcher[Job Matcher<br/>ML Pipeline]
            Recommender[Recommendation<br/>Engine]
            GraphProcessor[Graph Processor<br/>Connection Suggestions]
        end

        subgraph "Caching Layer"
            Redis_Profile[Redis<br/>Profile Cache]
            Redis_Feed[Redis<br/>Feed Cache]
            Redis_Connection[Redis<br/>Connection Cache]
        end

        subgraph "Storage Layer"
            UserDB[(User DB<br/>PostgreSQL)]
            GraphDB[(Social Graph<br/>Neo4j)]
            PostDB[(Posts<br/>Cassandra)]
            JobDB[(Jobs<br/>PostgreSQL)]
            SearchIndex[(Elasticsearch<br/>Search Index)]
            MessageDB[(Messages<br/>ScyllaDB)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event Stream]
        end

        subgraph "Object Storage"
            S3[S3<br/>Photos/Resumes]
        end

        Web --> Gateway
        Mobile --> Gateway

        Gateway --> Profile
        Gateway --> Connection
        Gateway --> Feed
        Gateway --> Job
        Gateway --> Search
        Gateway --> Message

        Profile --> UserDB
        Profile --> Redis_Profile
        Profile --> S3

        Connection --> GraphDB
        Connection --> Redis_Connection
        Connection --> Kafka

        Feed --> PostDB
        Feed --> Redis_Feed
        Feed --> Kafka

        Job --> JobDB
        Job --> JobMatcher

        Search --> SearchIndex

        Message --> MessageDB
        Message --> Notification

        Kafka --> FeedWorker
        Kafka --> JobMatcher
        Kafka --> Recommender
        Kafka --> GraphProcessor

        FeedWorker --> Redis_Feed
        JobMatcher --> Job
        Recommender --> Profile
        GraphProcessor --> GraphDB

        style Gateway fill:#e1f5ff
        style GraphDB fill:#ffe1e1
        style Redis_Feed fill:#fff4e1
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Technology | Why This? | Alternative |
    |-----------|-----------|-----------|-------------|
    | **Social Graph** | Neo4j | Graph queries (friends of friends), fast traversal | SQL (complex joins), custom graph (hard to maintain) |
    | **User Data** | PostgreSQL | ACID for profiles, complex queries | MongoDB (harder to join), DynamoDB (limited queries) |
    | **Posts** | Cassandra | High write throughput, time-series data | MySQL (write bottleneck), MongoDB (consistency issues) |
    | **Messages** | ScyllaDB | Low latency reads/writes, high throughput | Cassandra (slower), DynamoDB (expensive at scale) |
    | **Search** | Elasticsearch | Full-text search, faceted search, ranking | Solr (complex), custom search (reinvent wheel) |
    | **Cache** | Redis | Fast in-memory, pub/sub for real-time | Memcached (limited features), no cache (too slow) |
    | **Events** | Kafka | High throughput, replay capability | RabbitMQ (lower throughput), direct calls (not scalable) |

    ---

    ## API Design

    ### 1. Create/Update Profile

    **Request:**
    ```http
    PUT /api/v1/users/{user_id}/profile
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "headline": "Senior Software Engineer at Microsoft",
      "summary": "Passionate about distributed systems...",
      "experience": [
        {
          "company": "Microsoft",
          "title": "Senior Software Engineer",
          "start_date": "2020-01",
          "end_date": null,
          "description": "Building scalable cloud services"
        }
      ],
      "education": [...],
      "skills": ["Java", "Python", "Distributed Systems"],
      "profile_photo_url": "https://cdn.linkedin.com/photos/user123.jpg"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user123",
      "profile_url": "https://linkedin.com/in/johndoe",
      "updated_at": "2026-01-29T10:30:00Z"
    }
    ```

    ---

    ### 2. Send Connection Request

    **Request:**
    ```http
    POST /api/v1/connections/request
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "to_user_id": "user456",
      "message": "I'd love to connect with you!"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "connection_request_id": "req789",
      "status": "pending",
      "created_at": "2026-01-29T10:30:00Z"
    }
    ```

    ---

    ### 3. Get News Feed

    **Request:**
    ```http
    GET /api/v1/feed?cursor=xyz123&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "posts": [
        {
          "post_id": "post123",
          "author": {
            "user_id": "user456",
            "name": "Jane Smith",
            "headline": "Product Manager at Google",
            "profile_photo": "https://cdn.linkedin.com/photos/user456.jpg"
          },
          "content": "Excited to announce...",
          "created_at": "2026-01-29T09:00:00Z",
          "likes_count": 234,
          "comments_count": 45,
          "shares_count": 12,
          "media": [...]
        }
      ],
      "next_cursor": "xyz456",
      "has_more": true
    }
    ```

    ---

    ### 4. Search People

    **Request:**
    ```http
    GET /api/v1/search/people?q=software+engineer&location=San+Francisco&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "results": [
        {
          "user_id": "user789",
          "name": "John Doe",
          "headline": "Software Engineer at Meta",
          "location": "San Francisco, CA",
          "mutual_connections": 15,
          "profile_photo": "...",
          "connection_status": "2nd_degree"
        }
      ],
      "total_results": 1523,
      "next_cursor": "..."
    }
    ```

    ---

    ### 5. Post Job

    **Request:**
    ```http
    POST /api/v1/jobs
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "company_id": "comp123",
      "title": "Senior Software Engineer",
      "description": "We are looking for...",
      "location": "Seattle, WA",
      "employment_type": "Full-time",
      "experience_level": "Mid-Senior",
      "required_skills": ["Java", "Distributed Systems"],
      "salary_range": {
        "min": 150000,
        "max": 200000,
        "currency": "USD"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "job_id": "job456",
      "job_url": "https://linkedin.com/jobs/view/job456",
      "created_at": "2026-01-29T10:30:00Z",
      "expires_at": "2026-02-28T23:59:59Z"
    }
    ```

    ---

    ## Database Schema

    ### Users (PostgreSQL)

    ```sql
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        headline VARCHAR(200),
        summary TEXT,
        profile_photo_url TEXT,
        location VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_email (email),
        INDEX idx_location (location)
    );

    CREATE TABLE experience (
        experience_id BIGINT PRIMARY KEY,
        user_id BIGINT REFERENCES users(user_id),
        company_id BIGINT,
        title VARCHAR(200),
        start_date DATE,
        end_date DATE,
        description TEXT,
        INDEX idx_user_id (user_id)
    );

    CREATE TABLE skills (
        skill_id BIGINT PRIMARY KEY,
        user_id BIGINT REFERENCES users(user_id),
        skill_name VARCHAR(100),
        endorsement_count INT DEFAULT 0,
        INDEX idx_user_id (user_id),
        INDEX idx_skill_name (skill_name)
    );
    ```

    ---

    ### Social Graph (Neo4j)

    ```cypher
    // User node
    CREATE (u:User {
        user_id: 123,
        name: "John Doe"
    })

    // Connection relationship
    CREATE (u1:User {user_id: 123})-[:CONNECTED_TO {connected_at: 1643712000}]->(u2:User {user_id: 456})

    // Queries
    // Get 1st degree connections
    MATCH (me:User {user_id: 123})-[:CONNECTED_TO]-(friend:User)
    RETURN friend

    // Get 2nd degree connections (friends of friends)
    MATCH (me:User {user_id: 123})-[:CONNECTED_TO]-()-[:CONNECTED_TO]-(fof:User)
    WHERE NOT (me)-[:CONNECTED_TO]-(fof) AND me <> fof
    RETURN DISTINCT fof

    // People you may know (mutual connections)
    MATCH (me:User {user_id: 123})-[:CONNECTED_TO]-(mutual)-[:CONNECTED_TO]-(suggestion:User)
    WHERE NOT (me)-[:CONNECTED_TO]-(suggestion) AND me <> suggestion
    RETURN suggestion, COUNT(mutual) AS mutual_count
    ORDER BY mutual_count DESC
    LIMIT 10
    ```

    ---

    ### Posts (Cassandra)

    ```sql
    CREATE TABLE posts (
        post_id BIGINT PRIMARY KEY,
        user_id BIGINT,
        content TEXT,
        created_at TIMESTAMP,
        likes_count COUNTER,
        comments_count COUNTER,
        shares_count COUNTER,
        media_urls LIST<TEXT>
    );

    CREATE TABLE user_posts (
        user_id BIGINT,
        created_at TIMESTAMP,
        post_id BIGINT,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    CREATE TABLE feed (
        user_id BIGINT,
        created_at TIMESTAMP,
        post_id BIGINT,
        relevance_score DOUBLE,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);
    ```

    ---

    ### Jobs (PostgreSQL)

    ```sql
    CREATE TABLE jobs (
        job_id BIGINT PRIMARY KEY,
        company_id BIGINT,
        title VARCHAR(200),
        description TEXT,
        location VARCHAR(100),
        employment_type VARCHAR(50),
        experience_level VARCHAR(50),
        salary_min INT,
        salary_max INT,
        required_skills TEXT[],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        status VARCHAR(20),
        INDEX idx_company_id (company_id),
        INDEX idx_location (location),
        INDEX idx_status (status),
        INDEX idx_expires_at (expires_at)
    );

    CREATE TABLE job_applications (
        application_id BIGINT PRIMARY KEY,
        job_id BIGINT REFERENCES jobs(job_id),
        user_id BIGINT,
        resume_url TEXT,
        cover_letter TEXT,
        status VARCHAR(50),
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_job_id (job_id),
        INDEX idx_user_id (user_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Connection Request Flow

    ```mermaid
    sequenceDiagram
        participant User1
        participant API
        participant Connection_Service
        participant GraphDB
        participant Kafka
        participant Notification

        User1->>API: Send connection request
        API->>Connection_Service: Create connection request
        Connection_Service->>GraphDB: Check existing connection
        GraphDB-->>Connection_Service: No connection exists

        Connection_Service->>GraphDB: Create PENDING relationship
        GraphDB-->>Connection_Service: Success

        Connection_Service->>Kafka: Publish connection_request_sent event
        Connection_Service-->>User1: 201 Created

        Kafka->>Notification: Process event
        Notification->>User2: Send notification
    ```

    ### Feed Generation Flow

    ```mermaid
    sequenceDiagram
        participant User
        participant API
        participant Feed_Service
        participant Redis
        participant PostDB
        participant GraphDB

        User->>API: GET /feed
        API->>Feed_Service: Get personalized feed

        Feed_Service->>Redis: Check cached feed
        alt Cache HIT
            Redis-->>Feed_Service: Cached posts
        else Cache MISS
            Feed_Service->>GraphDB: Get connections
            GraphDB-->>Feed_Service: Connection IDs

            Feed_Service->>PostDB: Get recent posts from connections
            PostDB-->>Feed_Service: Posts

            Feed_Service->>Feed_Service: Rank posts by ML model
            Feed_Service->>Redis: Cache ranked feed (TTL: 5min)
        end

        Feed_Service-->>User: Personalized feed
    ```

=== "🔍 Step 3: Deep Dive"

    This section will be abbreviated for length. Key topics to cover:

    - **Social Graph:** Connection degrees, People You May Know algorithm
    - **Feed Generation:** Fan-out on write vs read, ML ranking
    - **Job Matching:** Skills matching, location preferences, ML recommendations
    - **Search:** Elasticsearch architecture, ranking by relevance
    - **Messaging:** Real-time chat, WebSocket connections, message persistence

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Current Capacity | Bottleneck? | Solution |
    |-----------|-----------------|------------|----------|
    | **Graph DB** | 10K graph queries/sec | 🟡 Yes | Read replicas, cache frequently accessed paths |
    | **Feed Generation** | 5K feeds/sec | ✅ Yes | Pre-compute feeds, Redis cache, async workers |
    | **Search** | 50K queries/sec | ❌ No | Elasticsearch cluster handles load |
    | **PostgreSQL** | 20K QPS | 🟡 Approaching | Sharding by user_id, read replicas |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Neo4j for social graph** - Fast connection queries
    2. **Hybrid feed generation** - Pre-compute + on-demand
    3. **ML-based job matching** - Skills + preferences
    4. **Elasticsearch for search** - Fast, relevant results
    5. **Redis caching** - Hot user data, feeds, connections
    6. **Eventual consistency** - Acceptable for social platform

    ## Interview Tips

    ✅ **Emphasize graph database** - Social connections are core
    ✅ **Discuss feed ranking** - ML-based personalization
    ✅ **Job matching complexity** - Skills, location, preferences
    ✅ **Search at scale** - Elasticsearch sharding
    ✅ **Real-time messaging** - WebSocket architecture

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** LinkedIn, Meta, Google, Amazon, Microsoft

---

*This problem combines social networking, job marketplace, and professional content feed - master this and you'll be ready for complex social platform designs.*
