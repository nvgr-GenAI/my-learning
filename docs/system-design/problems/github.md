# Design GitHub

Design a web-based version control platform that allows developers to host code repositories, collaborate through pull requests, track issues, review code, and manage software projects with features like CI/CD, wikis, and discussions.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M users, 400M repositories, 1B commits, 100M pull requests/year |
| **Key Challenges** | Git operations at scale, code search, merge conflicts, CI/CD orchestration, large file storage |
| **Core Concepts** | Distributed version control, blob storage, diff algorithms, merge strategies, webhook delivery |
| **Companies** | GitHub, GitLab, Bitbucket, Microsoft, Google, Amazon |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Repository Hosting** | Store and manage Git repositories | P0 (Must have) |
    | **Git Operations** | Push, pull, clone, commit, branch, merge | P0 (Must have) |
    | **Pull Requests** | Propose changes, code review, discussions | P0 (Must have) |
    | **Issues** | Bug tracking, feature requests | P0 (Must have) |
    | **Code Search** | Search code across repositories | P0 (Must have) |
    | **Web UI** | Browse code, view diffs, file history | P0 (Must have) |
    | **CI/CD (Actions)** | Automated build, test, deploy pipelines | P1 (Should have) |
    | **Webhooks** | Event notifications to external services | P1 (Should have) |
    | **Wiki** | Project documentation | P2 (Nice to have) |
    | **Projects** | Kanban boards for project management | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - GitHub Copilot (AI code completion)
    - GitHub Packages (package registry)
    - GitHub Discussions (forum)
    - GitHub Sponsors
    - Advanced security scanning
    - GitHub Enterprise Server (on-premises)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.95% uptime | Developers depend on it for work |
    | **Git Latency** | < 1s for push/pull | Fast operations critical |
    | **Code Search** | < 500ms | Quick search improves productivity |
    | **Data Durability** | No data loss | Code is developers' intellectual property |
    | **Scalability** | Millions of concurrent users | Global developer platform |
    | **Consistency** | Strong for Git, eventual for UI | Git integrity critical |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total users: 100 million
    Daily Active Users (DAU): 10 million
    Total repositories: 400 million
    Daily commits: 10 million commits
    Daily pull requests: 300,000 PRs

    Git operations per day:
    - Git push: 10M pushes/day
    - Git pull/clone: 50M pulls/day
    - Git fetch: 100M fetches/day

    QPS calculations:
    - Git push: 10M / 86,400 = 116 push/sec
    - Git pull: 50M / 86,400 = 579 pull/sec
    - Code search: 5M / 86,400 = 58 searches/sec
    - Web browsing: 50M / 86,400 = 579 page views/sec
    - API calls: 100M / 86,400 = 1,160 API/sec
    ```

    ### Storage Estimates

    ```
    Git repository data:
    - Average repo size: 100 MB (with full history)
    - 400M repos × 100 MB = 40 PB
    - With deduplication: ~25 PB (git objects are content-addressable)

    Large files (Git LFS):
    - 10M repos use LFS
    - Average LFS storage per repo: 500 MB
    - 10M × 500 MB = 5 PB

    Issues & Pull Requests:
    - 500M issues × 10 KB = 5 TB
    - 100M PRs × 50 KB = 5 TB

    CI/CD logs:
    - 1B workflow runs × 100 KB = 100 TB
    - With compression: 20 TB

    Total storage: 25 PB + 5 PB + 10 TB + 20 TB ≈ 30 PB
    ```

    ### Bandwidth Estimates

    ```
    Git operations:
    - Push (ingress): 116 push/sec × 10 MB = 1.16 GB/sec ≈ 9.3 Gbps
    - Pull (egress): 579 pull/sec × 10 MB = 5.79 GB/sec ≈ 46 Gbps

    Web UI:
    - Page views: 579/sec × 500 KB = 290 MB/sec ≈ 2.3 Gbps

    API:
    - API calls: 1,160/sec × 10 KB = 11.6 MB/sec ≈ 93 Mbps

    Total ingress: ~10 Gbps
    Total egress: ~50 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Git object cache:
    - Hot repositories (1M): 100 GB

    Repository metadata:
    - 10M active repos × 10 KB = 100 GB

    Code search index cache:
    - 50 GB

    Session data:
    - 1M concurrent users × 10 KB = 10 GB

    Total cache: 100 GB + 100 GB + 50 GB + 10 GB = 260 GB
    ```

    ---

    ## Key Assumptions

    1. Average repository size: 100 MB with full history
    2. 10% of repositories are actively developed (90% archived/inactive)
    3. Average 10 commits per push
    4. 80/20 rule: 20% of repos account for 80% of traffic
    5. Most operations are reads (pulls > pushes, 5:1 ratio)
    6. Peak hours during US/Europe business hours

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Git-native storage** - Use Git's object model (blobs, trees, commits)
    2. **Content-addressable** - Objects identified by SHA-1 hash
    3. **Distributed architecture** - Multiple storage clusters globally
    4. **Eventual consistency for UI** - Strong consistency for Git operations
    5. **Webhook-driven integrations** - Event-driven architecture

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Git_Client[Git Client<br/>CLI]
            Web[Web Browser]
            API_Client[API/CLI<br/>gh tool]
        end

        subgraph "Frontend"
            CDN[CDN<br/>Static Assets]
            WebServer[Web Servers<br/>Rails/React]
        end

        subgraph "API Layer"
            Git_API[Git Protocol<br/>Service]
            REST_API[REST API]
            GraphQL_API[GraphQL API]
        end

        subgraph "Service Layer"
            Repo_Service[Repository<br/>Service]
            PR_Service[Pull Request<br/>Service]
            Issue_Service[Issue<br/>Service]
            Search_Service[Search<br/>Service]
            Webhook_Service[Webhook<br/>Delivery]
            CI_Service[CI/CD Actions<br/>Orchestrator]
        end

        subgraph "Git Storage"
            Git_Storage[Git Object<br/>Storage]
            LFS_Storage[Large File<br/>Storage S3]
            Packs[Git Pack<br/>Files]
        end

        subgraph "Databases"
            Metadata_DB[(Repository<br/>Metadata MySQL)]
            Issue_DB[(Issues/PRs<br/>PostgreSQL)]
            Search_Index[(Code Search<br/>Elasticsearch)]
        end

        subgraph "Background Processing"
            IndexWorker[Code Indexer]
            WebhookWorker[Webhook Worker]
            MergeWorker[Merge Queue<br/>Worker]
            CI_Runner[CI/CD Runner<br/>Kubernetes]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event Bus]
        end

        Git_Client --> Git_API
        Web --> CDN
        Web --> WebServer
        API_Client --> REST_API
        API_Client --> GraphQL_API

        Git_API --> Repo_Service
        REST_API --> Repo_Service
        REST_API --> PR_Service
        REST_API --> Issue_Service
        REST_API --> Search_Service

        Repo_Service --> Git_Storage
        Repo_Service --> LFS_Storage
        Repo_Service --> Metadata_DB
        Repo_Service --> Kafka

        PR_Service --> Issue_DB
        PR_Service --> Kafka

        Issue_Service --> Issue_DB

        Search_Service --> Search_Index

        CI_Service --> CI_Runner
        CI_Service --> Kafka

        Webhook_Service --> Kafka

        Kafka --> IndexWorker
        Kafka --> WebhookWorker
        Kafka --> MergeWorker

        IndexWorker --> Search_Index
        WebhookWorker --> External[External<br/>Services]

        style Git_Storage fill:#ffe1e1
        style Search_Index fill:#fff4e1
        style Kafka fill:#e1f5ff
        style CI_Runner fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Technology | Why This? | Alternative |
    |-----------|-----------|-----------|-------------|
    | **Git Storage** | Custom (libgit2) + FS | Git-native, efficient, battle-tested | Cloud storage only (slow), database (wrong fit) |
    | **LFS Storage** | S3 | Scalable, durable, cost-effective for large files | Git repo (breaks with large files), custom (complex) |
    | **Metadata DB** | MySQL | Relational data, complex queries, ACID | NoSQL (harder queries), custom (complex) |
    | **Issues/PRs** | PostgreSQL | JSON support, full-text search, transactions | MySQL (similar), MongoDB (no transactions) |
    | **Code Search** | Elasticsearch | Fast full-text search, code-aware indexing | Database LIKE (too slow), custom (complex) |
    | **Event Bus** | Kafka | High throughput, replay, ordering guarantees | RabbitMQ (lower throughput), direct calls (not scalable) |
    | **CI/CD** | Kubernetes | Container orchestration, auto-scaling | VMs (less efficient), custom (complex) |

    ---

    ## API Design

    ### 1. Git Push (Git Protocol)

    **Request (Git protocol):**
    ```
    git push origin main

    POST /repos/user/repo/git-receive-pack
    Content-Type: application/x-git-receive-pack-request

    [Git packfile data]
    ```

    **Response:**
    ```
    HTTP/1.1 200 OK
    Content-Type: application/x-git-receive-pack-result

    [Git packfile result]
    ```

    ---

    ### 2. Create Pull Request

    **Request:**
    ```http
    POST /api/v1/repos/{owner}/{repo}/pulls
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "title": "Add user authentication",
      "head": "feature/auth",
      "base": "main",
      "body": "This PR implements user authentication using JWT...",
      "draft": false
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "id": 12345,
      "number": 42,
      "state": "open",
      "title": "Add user authentication",
      "user": {
        "login": "johndoe",
        "avatar_url": "..."
      },
      "head": {
        "ref": "feature/auth",
        "sha": "abc123..."
      },
      "base": {
        "ref": "main",
        "sha": "def456..."
      },
      "created_at": "2026-01-29T10:30:00Z",
      "html_url": "https://github.com/user/repo/pull/42"
    }
    ```

    ---

    ### 3. Search Code

    **Request:**
    ```http
    GET /api/v1/search/code
      ?q=function+authenticate
      &repo=user/repo
      &language=javascript
      &limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "total_count": 47,
      "items": [
        {
          "name": "auth.js",
          "path": "src/auth/auth.js",
          "sha": "abc123...",
          "repository": {
            "full_name": "user/repo"
          },
          "text_matches": [
            {
              "fragment": "function authenticate(user, password) {",
              "matches": [
                {
                  "text": "authenticate",
                  "indices": [9, 21]
                }
              ]
            }
          ],
          "html_url": "https://github.com/user/repo/blob/main/src/auth/auth.js"
        }
      ]
    }
    ```

    ---

    ### 4. Webhook Delivery

    **Webhook payload (POST to external URL):**
    ```json
    {
      "action": "opened",
      "pull_request": {
        "id": 12345,
        "number": 42,
        "title": "Add user authentication",
        "user": {...},
        "head": {...},
        "base": {...}
      },
      "repository": {
        "id": 789,
        "name": "repo",
        "full_name": "user/repo",
        "owner": {...}
      },
      "sender": {...}
    }
    ```

    ---

    ## Database Schema

    ### Repositories (MySQL)

    ```sql
    CREATE TABLE repositories (
        repo_id BIGINT PRIMARY KEY,
        owner_id BIGINT,
        name VARCHAR(100),
        full_name VARCHAR(200),
        description TEXT,
        is_private BOOLEAN DEFAULT FALSE,
        is_fork BOOLEAN DEFAULT FALSE,
        parent_repo_id BIGINT,
        default_branch VARCHAR(100) DEFAULT 'main',
        size_kb BIGINT,
        stars_count INT DEFAULT 0,
        forks_count INT DEFAULT 0,
        watchers_count INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        pushed_at TIMESTAMP,
        INDEX idx_owner (owner_id),
        INDEX idx_full_name (full_name),
        INDEX idx_stars (stars_count DESC)
    );
    ```

    ---

    ### Pull Requests (PostgreSQL)

    ```sql
    CREATE TABLE pull_requests (
        pr_id BIGINT PRIMARY KEY,
        repo_id BIGINT,
        number INT,
        title VARCHAR(500),
        body TEXT,
        state VARCHAR(20), -- open, closed, merged
        user_id BIGINT,
        head_branch VARCHAR(100),
        head_sha VARCHAR(40),
        base_branch VARCHAR(100),
        base_sha VARCHAR(40),
        merged_at TIMESTAMP,
        closed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_repo_state (repo_id, state),
        INDEX idx_user (user_id),
        UNIQUE (repo_id, number)
    );

    CREATE TABLE pr_reviews (
        review_id BIGINT PRIMARY KEY,
        pr_id BIGINT REFERENCES pull_requests(pr_id),
        user_id BIGINT,
        state VARCHAR(20), -- approved, changes_requested, commented
        body TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE pr_comments (
        comment_id BIGINT PRIMARY KEY,
        pr_id BIGINT REFERENCES pull_requests(pr_id),
        user_id BIGINT,
        body TEXT,
        path VARCHAR(500), -- file path
        line INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_pr (pr_id)
    );
    ```

    ---

    ### Git Objects (File System)

    Git objects are stored on disk using content-addressable storage:

    ```
    .git/objects/
    ├── ab/
    │   └── cdef123456... (blob)
    ├── cd/
    │   └── ef789012... (tree)
    ├── ef/
    │   └── 012345... (commit)
    └── pack/
        ├── pack-xyz.idx
        └── pack-xyz.pack
    ```

    **Object types:**
    - **Blob:** File contents
    - **Tree:** Directory listing (file names + blob refs)
    - **Commit:** Snapshot with metadata (author, message, parent)
    - **Tag:** Named reference to commit

    ---

    ## Data Flow Diagrams

    ### Git Push Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Git_API
        participant Repo_Service
        participant Git_Storage
        participant Kafka
        participant IndexWorker
        participant WebhookWorker

        Client->>Git_API: git push (packfile)
        Git_API->>Repo_Service: Validate & process push
        Repo_Service->>Repo_Service: Verify permissions
        Repo_Service->>Repo_Service: Receive objects
        Repo_Service->>Git_Storage: Write objects to disk
        Git_Storage-->>Repo_Service: Objects stored

        Repo_Service->>Repo_Service: Update refs (branches)
        Repo_Service->>Kafka: Publish push event
        Repo_Service-->>Client: Push success

        Kafka->>IndexWorker: Process push event
        IndexWorker->>Git_Storage: Read new commits
        IndexWorker->>IndexWorker: Parse code files
        IndexWorker->>Search_Index: Update search index

        Kafka->>WebhookWorker: Process push event
        WebhookWorker->>WebhookWorker: Get webhook subscriptions
        WebhookWorker->>External_Service: POST webhook payload
    ```

    ### Pull Request Merge Flow

    ```mermaid
    sequenceDiagram
        participant User
        participant Web
        participant PR_Service
        participant Git_Storage
        participant MergeWorker
        participant CI_Service

        User->>Web: Click "Merge PR"
        Web->>PR_Service: Merge request
        PR_Service->>PR_Service: Check CI status
        PR_Service->>PR_Service: Check reviews

        alt All checks pass
            PR_Service->>Git_Storage: Perform merge
            Git_Storage->>Git_Storage: Create merge commit
            Git_Storage->>Git_Storage: Update base branch
            Git_Storage-->>PR_Service: Merge commit SHA

            PR_Service->>PR_Service: Update PR state to "merged"
            PR_Service->>Kafka: Publish merge event
            PR_Service-->>User: Merge successful
        else Checks failed
            PR_Service-->>User: Cannot merge (status checks failed)
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics:

    ### 1. Git Object Storage
    - **Content-addressable:** Objects named by SHA-1 hash
    - **Deduplication:** Identical files stored once
    - **Pack files:** Compressed delta storage for efficiency
    - **Garbage collection:** Remove unreachable objects

    ### 2. Merge Conflict Resolution
    - **Three-way merge:** Base, head, and merge commits
    - **Conflict detection:** Overlapping changes
    - **Manual resolution:** User resolves conflicts
    - **Merge strategies:** Fast-forward, recursive, ours, theirs

    ### 3. Code Search Architecture
    - **Indexing pipeline:** Parse commits, extract code
    - **Language-aware:** Syntax highlighting, symbol extraction
    - **Ranking:** Relevance, popularity, recency
    - **Filters:** Language, repository, path

    ### 4. CI/CD Orchestration
    - **Workflow definition:** YAML files in `.github/workflows/`
    - **Trigger events:** Push, PR, schedule, manual
    - **Runner allocation:** Kubernetes pods for isolation
    - **Artifact storage:** S3 for build outputs

    ### 5. Webhook Delivery
    - **Event types:** Push, PR, issue, release
    - **Retry logic:** Exponential backoff
    - **Security:** HMAC signature verification
    - **Delivery guarantees:** At-least-once

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Git Storage** | ✅ Yes | Sharding by repository, replicas, CDN for popular repos |
    | **Code Search** | 🟡 Maybe | Elasticsearch cluster, sharding, incremental indexing |
    | **Database** | ❌ No | Read replicas, caching, sharding if needed |
    | **CI/CD** | ✅ Yes | Kubernetes auto-scaling, spot instances, queue management |

    ## Performance Optimization

    - **Git protocol optimization:** Pack files, shallow clones
    - **CDN for popular repos:** Cache frequently cloned repos
    - **Lazy indexing:** Index popular repos first
    - **Database connection pooling:** Reuse connections
    - **Caching:** Redis for metadata, hot repos

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Git-native storage** - Use Git's proven data model
    2. **Content-addressable** - Efficient deduplication
    3. **Elasticsearch for search** - Fast, code-aware
    4. **Kafka for events** - Reliable, replay capability
    5. **Kubernetes for CI/CD** - Scalable, isolated runners
    6. **S3 for LFS** - Cost-effective large file storage

    ## Interview Tips

    ✅ **Understand Git internals** - Objects, refs, pack files
    ✅ **Code search complexity** - Language-aware indexing
    ✅ **Merge strategies** - Three-way merge, conflicts
    ✅ **CI/CD orchestration** - Workflow triggers, runners
    ✅ **Webhook reliability** - Retry logic, at-least-once

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How does Git store data?"** | Content-addressable, blobs/trees/commits, pack files for compression |
    | **"How to handle merge conflicts?"** | Three-way merge, detect overlapping changes, manual resolution |
    | **"How to scale code search?"** | Elasticsearch cluster, sharding, incremental indexing, language-aware |
    | **"How to ensure webhook delivery?"** | Kafka for durability, retry with exponential backoff, dead letter queue |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** GitHub, GitLab, Bitbucket, Microsoft, Google, Amazon

---

*Master this problem and you'll understand version control systems, code hosting, and developer tooling platforms.*
