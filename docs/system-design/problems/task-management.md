# Design a Task Management System

Design a collaborative task management platform like Asana, Jira, or Trello where teams can create tasks, organize them into projects and boards, track progress, set dependencies, automate workflows, and collaborate in real-time with notifications and activity feeds.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐ Medium | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M users, 5M DAU, 500M tasks, 100K concurrent users |
| **Key Challenges** | Task dependencies, workflow automation, real-time collaboration, notification delivery, activity tracking |
| **Core Concepts** | DAG for dependencies, state machines, WebSocket for real-time, event sourcing, search indexing |
| **Companies** | Asana, Jira, Trello, Monday.com, ClickUp, Linear |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Task CRUD** | Create, read, update, delete tasks | P0 (Must have) |
    | **Task Assignment** | Assign tasks to users, change assignee | P0 (Must have) |
    | **Status Management** | Todo, In Progress, Done, Custom statuses | P0 (Must have) |
    | **Projects & Boards** | Group tasks into projects, Kanban boards | P0 (Must have) |
    | **Task Dependencies** | Block/blocked-by relationships | P0 (Must have) |
    | **Subtasks** | Hierarchical task breakdown | P0 (Must have) |
    | **Due Dates & Priorities** | Set deadlines, priority levels (P0-P2) | P0 (Must have) |
    | **Comments & Mentions** | Discuss tasks, @mention users | P1 (Should have) |
    | **Real-time Updates** | See changes instantly | P1 (Should have) |
    | **Notifications** | Email, push, in-app notifications | P1 (Should have) |
    | **Activity Feed** | Audit log of all changes | P1 (Should have) |
    | **Search & Filter** | Find tasks by various criteria | P1 (Should have) |
    | **Workflow Automation** | Auto-assign, auto-status on conditions | P2 (Nice to have) |
    | **Tags & Labels** | Categorize tasks | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Time tracking (hours logged)
    - Gantt charts and timeline views
    - Resource management
    - Budget tracking
    - Advanced reporting and analytics
    - Third-party integrations (Slack, GitHub)
    - File attachments (simplified)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Teams rely on it for work coordination |
    | **Latency** | < 200ms for API calls | Responsive UI is critical |
    | **Real-time Updates** | < 1s propagation | Collaboration requires instant feedback |
    | **Consistency** | Eventual consistency | Users can tolerate slight delay in updates |
    | **Scalability** | 5M DAU | Handle growing teams and enterprises |
    | **Data Durability** | No data loss | Task data is critical business information |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total users: 50M registered users
    Daily Active Users (DAU): 5M users (10% of registered)
    Concurrent users: 100K peak

    Task operations per day:
    - Create tasks: 10M/day (2 per active user)
    - Update tasks: 20M/day (4 per active user)
    - Read tasks: 100M/day (20 per active user)
    - Comments: 5M/day (1 per active user)

    QPS calculations:
    - Task create: 10M / 86,400 = 116 creates/sec
    - Task update: 20M / 86,400 = 231 updates/sec
    - Task read: 100M / 86,400 = 1,157 reads/sec
    - Comment create: 5M / 86,400 = 58 comments/sec
    - Peak (3x): 347/694/3,471/174 per sec

    Real-time connections:
    - 100K concurrent WebSocket connections
    - Each user in ~5 projects (subscribed to updates)
    - Broadcast events: 231 updates/sec × 5 users = 1,155 events/sec
    ```

    ### Storage Estimates

    ```
    Tasks:
    - Total tasks: 500M tasks
    - Task size: 2 KB (title, description, metadata)
    - Total: 500M × 2 KB = 1 TB

    Comments:
    - Average 5 comments per task
    - Comment size: 500 bytes
    - Total: 500M × 5 × 500 bytes = 1.25 TB

    Activity logs:
    - 10 events per task (average lifecycle)
    - Event size: 300 bytes
    - Total: 500M × 10 × 300 bytes = 1.5 TB

    Users:
    - 50M users × 5 KB = 250 GB

    Projects:
    - 10M projects × 10 KB = 100 GB

    Total storage: 1 TB + 1.25 TB + 1.5 TB + 250 GB + 100 GB ≈ 4 TB
    ```

    ### Bandwidth Estimates

    ```
    API requests:
    - Read: 1,157 req/sec × 2 KB = 2.3 MB/sec ≈ 18 Mbps
    - Write: 347 req/sec × 2 KB = 694 KB/sec ≈ 5.5 Mbps

    Real-time updates (WebSocket):
    - 1,155 events/sec × 500 bytes = 577 KB/sec ≈ 4.6 Mbps

    Notifications:
    - 2,000 notifications/sec × 300 bytes = 600 KB/sec ≈ 4.8 Mbps

    Total ingress: ~6 Mbps
    Total egress: ~28 Mbps
    ```

    ### Memory Estimates (Caching)

    ```
    Active tasks cache:
    - 10M active tasks × 2 KB = 20 GB

    User sessions:
    - 100K concurrent users × 10 KB = 1 GB

    Project metadata:
    - 1M active projects × 10 KB = 10 GB

    WebSocket connection state:
    - 100K connections × 5 KB = 500 MB

    Total cache: 20 GB + 1 GB + 10 GB + 0.5 GB ≈ 32 GB
    ```

    ---

    ## Key Assumptions

    1. Average task has 5 comments over its lifetime
    2. Average user is in 3-5 projects
    3. Average project has 100-500 tasks
    4. 60% of tasks are completed, 40% active
    5. Peak hours during business hours (9am-5pm across timezones)
    6. Task dependencies: 10% of tasks have dependencies
    7. Subtasks: 20% of tasks have 2-3 subtasks

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Event-driven architecture** - All changes emit events for real-time updates
    2. **DAG for dependencies** - Detect cycles, validate dependency chains
    3. **State machine for workflows** - Define valid status transitions
    4. **WebSocket for real-time** - Push updates to connected clients
    5. **CQRS pattern** - Separate read and write models for scalability

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web App<br/>React]
            Mobile[Mobile App<br/>iOS/Android]
        end

        subgraph "Load Balancing"
            LB[Load Balancer<br/>ALB]
            WS_LB[WebSocket LB<br/>Sticky Sessions]
        end

        subgraph "Application Layer"
            API[REST API<br/>Task CRUD]
            WS[WebSocket Server<br/>Real-time Updates]
            Search_API[Search Service]
            Workflow_Engine[Workflow Engine<br/>Automation]
        end

        subgraph "Service Layer"
            Task_Service[Task Service]
            Project_Service[Project Service]
            User_Service[User Service]
            Comment_Service[Comment Service]
            Activity_Service[Activity Service]
            Notification_Service[Notification Service]
            Dependency_Service[Dependency Service]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event Bus]
        end

        subgraph "Background Workers"
            NotificationWorker[Notification<br/>Worker]
            SearchIndexer[Search<br/>Indexer]
            WorkflowWorker[Workflow<br/>Worker]
            ActivityLogger[Activity<br/>Logger]
        end

        subgraph "Storage Layer"
            TaskDB[(PostgreSQL<br/>Tasks, Projects)]
            Cache[(Redis<br/>Cache, Sessions)]
            SearchIndex[(Elasticsearch<br/>Search Index)]
            ActivityStore[(Cassandra<br/>Activity Log)]
        end

        subgraph "External Services"
            Email[Email Service<br/>SendGrid]
            Push[Push Notifications<br/>FCM/APNS]
        end

        Web --> LB
        Mobile --> LB
        Web --> WS_LB
        Mobile --> WS_LB

        LB --> API
        LB --> Search_API
        WS_LB --> WS

        API --> Task_Service
        API --> Project_Service
        API --> User_Service
        API --> Comment_Service

        Task_Service --> TaskDB
        Task_Service --> Cache
        Task_Service --> Kafka
        Task_Service --> Dependency_Service

        Dependency_Service --> TaskDB

        Project_Service --> TaskDB
        User_Service --> TaskDB
        Comment_Service --> TaskDB
        Comment_Service --> Kafka

        Search_API --> SearchIndex

        WS --> Cache

        Kafka --> NotificationWorker
        Kafka --> SearchIndexer
        Kafka --> WorkflowWorker
        Kafka --> ActivityLogger
        Kafka --> WS

        NotificationWorker --> Notification_Service
        Notification_Service --> Email
        Notification_Service --> Push

        SearchIndexer --> SearchIndex
        ActivityLogger --> ActivityStore

        WorkflowWorker --> Workflow_Engine
        Workflow_Engine --> Task_Service

        style TaskDB fill:#e1f5ff
        style Cache fill:#ffe1e1
        style Kafka fill:#fff4e1
        style SearchIndex fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Technology | Why This? | Alternative |
    |-----------|-----------|-----------|-------------|
    | **Primary DB** | PostgreSQL | ACID transactions, complex queries, foreign keys | MySQL (similar), MongoDB (no transactions) |
    | **Cache** | Redis | Fast reads, session storage, pub/sub | Memcached (no pub/sub), in-memory (not distributed) |
    | **Event Bus** | Kafka | High throughput, replay, ordering | RabbitMQ (lower throughput), SQS (higher latency) |
    | **Search** | Elasticsearch | Full-text search, filtering, aggregations | PostgreSQL full-text (slower), Algolia (expensive) |
    | **Activity Log** | Cassandra | Time-series data, high write throughput | PostgreSQL (slower writes), MongoDB (less optimized) |
    | **Real-time** | WebSocket | Bidirectional, low latency | HTTP polling (wasteful), SSE (one-way only) |

    ---

    ## API Design

    ### 1. Create Task

    **Request:**
    ```http
    POST /api/v1/tasks
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "project_id": "proj_123",
      "title": "Implement user authentication",
      "description": "Add JWT-based authentication to the API",
      "assignee_id": "user_456",
      "status": "todo",
      "priority": "high",
      "due_date": "2026-02-15T17:00:00Z",
      "tags": ["backend", "security"],
      "parent_task_id": null
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "task_id": "task_789",
      "project_id": "proj_123",
      "title": "Implement user authentication",
      "description": "Add JWT-based authentication to the API",
      "assignee": {
        "user_id": "user_456",
        "name": "Jane Doe",
        "avatar_url": "https://..."
      },
      "creator": {
        "user_id": "user_123",
        "name": "John Smith"
      },
      "status": "todo",
      "priority": "high",
      "due_date": "2026-02-15T17:00:00Z",
      "tags": ["backend", "security"],
      "created_at": "2026-02-05T10:30:00Z",
      "updated_at": "2026-02-05T10:30:00Z"
    }
    ```

    ---

    ### 2. Update Task

    **Request:**
    ```http
    PATCH /api/v1/tasks/task_789
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "status": "in_progress",
      "assignee_id": "user_789"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "task_id": "task_789",
      "status": "in_progress",
      "assignee": {
        "user_id": "user_789",
        "name": "Bob Johnson"
      },
      "updated_at": "2026-02-05T11:00:00Z",
      "updated_by": {
        "user_id": "user_456",
        "name": "Jane Doe"
      }
    }
    ```

    ---

    ### 3. Add Task Dependency

    **Request:**
    ```http
    POST /api/v1/tasks/task_789/dependencies
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "blocks_task_id": "task_999",
      "dependency_type": "blocks"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "dependency_id": "dep_123",
      "from_task_id": "task_789",
      "to_task_id": "task_999",
      "type": "blocks",
      "created_at": "2026-02-05T11:15:00Z"
    }
    ```

    ---

    ### 4. Add Comment

    **Request:**
    ```http
    POST /api/v1/tasks/task_789/comments
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "text": "I've started working on this. @user_123 can you review the design doc?",
      "mentions": ["user_123"]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "comment_id": "cmt_456",
      "task_id": "task_789",
      "author": {
        "user_id": "user_456",
        "name": "Jane Doe"
      },
      "text": "I've started working on this. @user_123 can you review the design doc?",
      "mentions": [
        {
          "user_id": "user_123",
          "name": "John Smith"
        }
      ],
      "created_at": "2026-02-05T11:30:00Z"
    }
    ```

    ---

    ### 5. Search Tasks

    **Request:**
    ```http
    GET /api/v1/tasks/search
      ?q=authentication
      &project_id=proj_123
      &status=in_progress
      &assignee_id=user_456
      &sort=due_date
      &limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tasks": [
        {
          "task_id": "task_789",
          "title": "Implement user authentication",
          "description": "Add JWT-based authentication...",
          "status": "in_progress",
          "priority": "high",
          "assignee": {...},
          "due_date": "2026-02-15T17:00:00Z",
          "highlight": "Implement user <em>authentication</em>"
        }
      ],
      "total": 1,
      "page": 1,
      "has_more": false
    }
    ```

    ---

    ### 6. Get Activity Feed

    **Request:**
    ```http
    GET /api/v1/tasks/task_789/activity
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "activities": [
        {
          "activity_id": "act_001",
          "task_id": "task_789",
          "actor": {
            "user_id": "user_456",
            "name": "Jane Doe"
          },
          "action": "status_changed",
          "changes": {
            "field": "status",
            "old_value": "todo",
            "new_value": "in_progress"
          },
          "timestamp": "2026-02-05T11:00:00Z"
        },
        {
          "activity_id": "act_002",
          "task_id": "task_789",
          "actor": {
            "user_id": "user_123",
            "name": "John Smith"
          },
          "action": "task_created",
          "timestamp": "2026-02-05T10:30:00Z"
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Tasks Table (PostgreSQL)

    ```sql
    CREATE TABLE tasks (
        task_id BIGINT PRIMARY KEY,
        project_id BIGINT NOT NULL REFERENCES projects(project_id),
        parent_task_id BIGINT REFERENCES tasks(task_id),
        title VARCHAR(500) NOT NULL,
        description TEXT,
        status VARCHAR(50) NOT NULL DEFAULT 'todo',
        priority VARCHAR(20) NOT NULL DEFAULT 'medium',
        assignee_id BIGINT REFERENCES users(user_id),
        creator_id BIGINT NOT NULL REFERENCES users(user_id),
        due_date TIMESTAMP,
        completed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_project (project_id),
        INDEX idx_assignee (assignee_id),
        INDEX idx_status (status),
        INDEX idx_due_date (due_date),
        INDEX idx_parent (parent_task_id)
    );

    CREATE TABLE task_tags (
        task_id BIGINT REFERENCES tasks(task_id) ON DELETE CASCADE,
        tag VARCHAR(50) NOT NULL,
        PRIMARY KEY (task_id, tag),
        INDEX idx_tag (tag)
    );
    ```

    ---

    ### Task Dependencies Table

    ```sql
    CREATE TABLE task_dependencies (
        dependency_id BIGINT PRIMARY KEY,
        from_task_id BIGINT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        to_task_id BIGINT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        dependency_type VARCHAR(20) NOT NULL, -- 'blocks', 'blocked_by'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (from_task_id, to_task_id),
        INDEX idx_from_task (from_task_id),
        INDEX idx_to_task (to_task_id),
        CHECK (from_task_id != to_task_id)
    );
    ```

    ---

    ### Projects Table

    ```sql
    CREATE TABLE projects (
        project_id BIGINT PRIMARY KEY,
        workspace_id BIGINT NOT NULL REFERENCES workspaces(workspace_id),
        name VARCHAR(200) NOT NULL,
        description TEXT,
        view_type VARCHAR(20) DEFAULT 'list', -- 'list', 'board', 'timeline'
        is_private BOOLEAN DEFAULT FALSE,
        owner_id BIGINT NOT NULL REFERENCES users(user_id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_workspace (workspace_id),
        INDEX idx_owner (owner_id)
    );

    CREATE TABLE project_members (
        project_id BIGINT REFERENCES projects(project_id) ON DELETE CASCADE,
        user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL, -- 'admin', 'member', 'viewer'
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (project_id, user_id)
    );
    ```

    ---

    ### Comments Table

    ```sql
    CREATE TABLE comments (
        comment_id BIGINT PRIMARY KEY,
        task_id BIGINT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        author_id BIGINT NOT NULL REFERENCES users(user_id),
        text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_task (task_id),
        INDEX idx_author (author_id)
    );

    CREATE TABLE comment_mentions (
        comment_id BIGINT REFERENCES comments(comment_id) ON DELETE CASCADE,
        user_id BIGINT REFERENCES users(user_id),
        PRIMARY KEY (comment_id, user_id),
        INDEX idx_user (user_id)
    );
    ```

    ---

    ### Activity Log (Cassandra)

    ```sql
    CREATE TABLE activities (
        task_id BIGINT,
        activity_id TIMEUUID,
        actor_id BIGINT,
        action VARCHAR,
        entity_type VARCHAR,
        entity_id BIGINT,
        changes MAP<TEXT, TEXT>,
        timestamp TIMESTAMP,
        PRIMARY KEY (task_id, activity_id)
    ) WITH CLUSTERING ORDER BY (activity_id DESC);

    CREATE TABLE user_activities (
        user_id BIGINT,
        activity_id TIMEUUID,
        task_id BIGINT,
        action VARCHAR,
        timestamp TIMESTAMP,
        PRIMARY KEY (user_id, activity_id)
    ) WITH CLUSTERING ORDER BY (activity_id DESC);
    ```

    ---

    ## Data Flow Diagrams

    ### Task Update Flow with Real-time

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Task_Service
        participant DB
        participant Kafka
        participant WS_Server
        participant Redis
        participant ActivityLogger

        Client->>API: PATCH /tasks/{id}
        API->>Task_Service: Update task
        Task_Service->>DB: BEGIN TRANSACTION
        Task_Service->>DB: UPDATE tasks SET status='in_progress'
        Task_Service->>DB: COMMIT
        DB-->>Task_Service: Success

        Task_Service->>Kafka: Publish task.updated event
        Task_Service->>Redis: Invalidate cache
        Task_Service-->>Client: 200 OK

        par Async Processing
            Kafka->>WS_Server: task.updated event
            WS_Server->>Redis: Get subscribed users
            Redis-->>WS_Server: [user_123, user_456, ...]

            loop For each subscribed user
                WS_Server->>Client: Push update via WebSocket
            end
        and
            Kafka->>ActivityLogger: task.updated event
            ActivityLogger->>ActivityStore: INSERT activity log
        and
            Kafka->>SearchIndexer: task.updated event
            SearchIndexer->>SearchIndex: Update document
        end
    ```

    ---

    ### Dependency Validation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Dependency_Service
        participant DB

        Client->>API: POST /tasks/{id}/dependencies
        API->>Dependency_Service: Add dependency

        Dependency_Service->>Dependency_Service: Check if creates cycle
        Dependency_Service->>DB: Query task_dependencies
        DB-->>Dependency_Service: Existing dependencies

        Dependency_Service->>Dependency_Service: Run DFS to detect cycle

        alt Cycle detected
            Dependency_Service-->>API: 400 Bad Request (cycle detected)
            API-->>Client: Error: Would create circular dependency
        else No cycle
            Dependency_Service->>DB: INSERT INTO task_dependencies
            Dependency_Service->>Kafka: Publish dependency.created
            Dependency_Service-->>API: 201 Created
            API-->>Client: Success
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics to Cover

    ### 1. Task Dependency Management (DAG)

    **Directed Acyclic Graph (DAG) for dependencies:**

    ```python
    from collections import defaultdict, deque
    from typing import List, Set

    class DependencyManager:
        def __init__(self, db):
            self.db = db

        def add_dependency(self, from_task_id: int, to_task_id: int) -> bool:
            """
            Add dependency: from_task blocks to_task
            Returns False if it would create a cycle
            """
            if self._would_create_cycle(from_task_id, to_task_id):
                return False

            self.db.execute("""
                INSERT INTO task_dependencies (from_task_id, to_task_id, dependency_type)
                VALUES (?, ?, 'blocks')
            """, from_task_id, to_task_id)

            return True

        def _would_create_cycle(self, from_task_id: int, to_task_id: int) -> bool:
            """
            Check if adding this dependency would create a cycle
            Use DFS to detect cycle
            """
            # Build adjacency list of current dependencies
            graph = self._build_dependency_graph()

            # Add the proposed edge
            graph[from_task_id].append(to_task_id)

            # Check if we can reach from_task_id starting from to_task_id
            # If yes, adding this edge creates a cycle
            visited = set()
            return self._has_path(graph, to_task_id, from_task_id, visited)

        def _build_dependency_graph(self) -> dict:
            """Build adjacency list from database"""
            dependencies = self.db.query("""
                SELECT from_task_id, to_task_id
                FROM task_dependencies
            """)

            graph = defaultdict(list)
            for dep in dependencies:
                graph[dep.from_task_id].append(dep.to_task_id)

            return graph

        def _has_path(self, graph: dict, start: int, end: int,
                     visited: Set[int]) -> bool:
            """DFS to check if path exists from start to end"""
            if start == end:
                return True

            visited.add(start)

            for neighbor in graph.get(start, []):
                if neighbor not in visited:
                    if self._has_path(graph, neighbor, end, visited):
                        return True

            return False

        def get_blocked_tasks(self, task_id: int) -> List[int]:
            """Get all tasks blocked by this task"""
            return self.db.query("""
                SELECT to_task_id
                FROM task_dependencies
                WHERE from_task_id = ?
            """, task_id)

        def get_blocking_tasks(self, task_id: int) -> List[int]:
            """Get all tasks blocking this task"""
            return self.db.query("""
                SELECT from_task_id
                FROM task_dependencies
                WHERE to_task_id = ?
            """, task_id)

        def topological_sort(self, task_ids: List[int]) -> List[int]:
            """
            Return tasks in dependency order (Kahn's algorithm)
            Useful for displaying tasks in execution order
            """
            graph = self._build_dependency_graph()
            in_degree = defaultdict(int)

            # Calculate in-degrees
            for task_id in task_ids:
                if task_id not in in_degree:
                    in_degree[task_id] = 0
                for neighbor in graph[task_id]:
                    in_degree[neighbor] += 1

            # BFS with queue of zero in-degree nodes
            queue = deque([tid for tid in task_ids if in_degree[tid] == 0])
            result = []

            while queue:
                task_id = queue.popleft()
                result.append(task_id)

                for neighbor in graph[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return result
    ```

    ---

    ### 2. Workflow Automation Engine

    **State machine for task status transitions:**

    ```python
    from enum import Enum
    from typing import Dict, List, Optional

    class TaskStatus(Enum):
        TODO = "todo"
        IN_PROGRESS = "in_progress"
        IN_REVIEW = "in_review"
        DONE = "done"
        BLOCKED = "blocked"
        CANCELLED = "cancelled"

    class WorkflowEngine:
        def __init__(self):
            # Define valid state transitions
            self.transitions = {
                TaskStatus.TODO: [TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED],
                TaskStatus.IN_PROGRESS: [TaskStatus.IN_REVIEW, TaskStatus.BLOCKED,
                                        TaskStatus.TODO, TaskStatus.CANCELLED],
                TaskStatus.IN_REVIEW: [TaskStatus.DONE, TaskStatus.IN_PROGRESS,
                                      TaskStatus.CANCELLED],
                TaskStatus.BLOCKED: [TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED],
                TaskStatus.DONE: [],
                TaskStatus.CANCELLED: [TaskStatus.TODO]
            }

            # Automation rules
            self.rules = []

        def is_valid_transition(self, from_status: TaskStatus,
                               to_status: TaskStatus) -> bool:
            """Check if status transition is allowed"""
            return to_status in self.transitions.get(from_status, [])

        def add_rule(self, rule: 'AutomationRule'):
            """Add workflow automation rule"""
            self.rules.append(rule)

        def apply_rules(self, task: dict, event: str):
            """Apply automation rules when event occurs"""
            for rule in self.rules:
                if rule.matches(task, event):
                    rule.execute(task)

    class AutomationRule:
        def __init__(self, name: str, trigger: dict, conditions: List[dict],
                     actions: List[dict]):
            self.name = name
            self.trigger = trigger  # {"event": "status_changed", "to": "in_review"}
            self.conditions = conditions  # [{"field": "priority", "operator": "eq", "value": "high"}]
            self.actions = actions  # [{"type": "assign", "user_id": "user_123"}]

        def matches(self, task: dict, event: str) -> bool:
            """Check if rule should be triggered"""
            # Check trigger
            if event != self.trigger.get("event"):
                return False

            # Check conditions
            for condition in self.conditions:
                field = condition["field"]
                operator = condition["operator"]
                expected = condition["value"]
                actual = task.get(field)

                if operator == "eq" and actual != expected:
                    return False
                elif operator == "neq" and actual == expected:
                    return False
                elif operator == "in" and actual not in expected:
                    return False

            return True

        def execute(self, task: dict):
            """Execute rule actions"""
            for action in self.actions:
                action_type = action["type"]

                if action_type == "assign":
                    # Auto-assign to user
                    task_service.update_task(
                        task["task_id"],
                        {"assignee_id": action["user_id"]}
                    )
                elif action_type == "set_status":
                    # Auto-change status
                    task_service.update_task(
                        task["task_id"],
                        {"status": action["status"]}
                    )
                elif action_type == "add_comment":
                    # Auto-add comment
                    comment_service.create_comment(
                        task["task_id"],
                        {"text": action["text"], "is_automated": True}
                    )
                elif action_type == "send_notification":
                    # Send notification
                    notification_service.send(
                        user_id=action["user_id"],
                        message=action["message"]
                    )

    # Example usage:
    workflow = WorkflowEngine()

    # Rule: When high-priority task moves to "in_review", auto-assign to tech lead
    workflow.add_rule(AutomationRule(
        name="Auto-assign high-priority reviews",
        trigger={"event": "status_changed", "to": "in_review"},
        conditions=[{"field": "priority", "operator": "eq", "value": "high"}],
        actions=[{"type": "assign", "user_id": "tech_lead_123"}]
    ))

    # Rule: When task completed, update parent task progress
    workflow.add_rule(AutomationRule(
        name="Update parent on subtask completion",
        trigger={"event": "status_changed", "to": "done"},
        conditions=[{"field": "parent_task_id", "operator": "neq", "value": None}],
        actions=[{"type": "recalculate_parent_progress"}]
    ))
    ```

    ---

    ### 3. Real-time Collaboration (WebSocket)

    **WebSocket server for real-time updates:**

    ```python
    import asyncio
    import json
    from typing import Dict, Set
    import websockets

    class WebSocketManager:
        def __init__(self):
            # user_id -> set of WebSocket connections
            self.connections: Dict[int, Set] = {}

            # user_id -> set of subscribed project_ids
            self.subscriptions: Dict[int, Set[int]] = {}

        async def register(self, websocket, user_id: int, project_ids: List[int]):
            """Register user connection and subscriptions"""
            if user_id not in self.connections:
                self.connections[user_id] = set()

            self.connections[user_id].add(websocket)
            self.subscriptions[user_id] = set(project_ids)

            print(f"User {user_id} connected, subscribed to {project_ids}")

        async def unregister(self, websocket, user_id: int):
            """Remove user connection"""
            if user_id in self.connections:
                self.connections[user_id].discard(websocket)
                if not self.connections[user_id]:
                    del self.connections[user_id]
                    del self.subscriptions[user_id]

        async def broadcast_to_project(self, project_id: int, event: dict):
            """Broadcast event to all users subscribed to project"""
            message = json.dumps(event)

            for user_id, subscribed_projects in self.subscriptions.items():
                if project_id in subscribed_projects:
                    # Send to all connections for this user
                    for websocket in self.connections.get(user_id, []):
                        try:
                            await websocket.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            await self.unregister(websocket, user_id)

        async def send_to_user(self, user_id: int, event: dict):
            """Send event to specific user"""
            message = json.dumps(event)

            for websocket in self.connections.get(user_id, []):
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    await self.unregister(websocket, user_id)

        async def handle_connection(self, websocket, path):
            """Handle WebSocket connection"""
            user_id = None

            try:
                # Authenticate and get subscriptions
                auth_message = await websocket.recv()
                auth_data = json.loads(auth_message)

                user_id = auth_data["user_id"]
                project_ids = auth_data["project_ids"]

                await self.register(websocket, user_id, project_ids)

                # Keep connection alive
                async for message in websocket:
                    # Handle ping/pong
                    if message == "ping":
                        await websocket.send("pong")

            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if user_id:
                    await self.unregister(websocket, user_id)

    # Kafka consumer to forward events to WebSocket
    async def kafka_to_websocket(ws_manager: WebSocketManager):
        """Consume Kafka events and broadcast via WebSocket"""
        consumer = KafkaConsumer('task-events')

        for message in consumer:
            event = json.loads(message.value)
            event_type = event["type"]

            if event_type in ["task.created", "task.updated", "task.deleted"]:
                project_id = event["data"]["project_id"]
                await ws_manager.broadcast_to_project(project_id, event)

            elif event_type == "task.commented":
                # Notify task assignee and mentioned users
                assignee_id = event["data"]["assignee_id"]
                mentioned_ids = event["data"]["mentioned_user_ids"]

                for user_id in [assignee_id] + mentioned_ids:
                    await ws_manager.send_to_user(user_id, event)
    ```

    ---

    ### 4. Notification System

    **Multi-channel notification delivery:**

    ```python
    from enum import Enum
    from typing import List

    class NotificationType(Enum):
        TASK_ASSIGNED = "task_assigned"
        TASK_COMMENTED = "task_commented"
        TASK_MENTIONED = "task_mentioned"
        TASK_DUE_SOON = "task_due_soon"
        TASK_COMPLETED = "task_completed"

    class NotificationChannel(Enum):
        IN_APP = "in_app"
        EMAIL = "email"
        PUSH = "push"

    class NotificationService:
        def __init__(self, email_service, push_service, db):
            self.email_service = email_service
            self.push_service = push_service
            self.db = db

        def send_notification(self, user_id: int, notification_type: NotificationType,
                             data: dict):
            """Send notification through appropriate channels"""
            # Get user notification preferences
            preferences = self.db.query("""
                SELECT channel, enabled
                FROM user_notification_preferences
                WHERE user_id = ? AND notification_type = ?
            """, user_id, notification_type.value)

            enabled_channels = {pref.channel for pref in preferences if pref.enabled}

            # In-app notification (always create)
            self._create_in_app_notification(user_id, notification_type, data)

            # Email notification
            if NotificationChannel.EMAIL.value in enabled_channels:
                self._send_email_notification(user_id, notification_type, data)

            # Push notification
            if NotificationChannel.PUSH.value in enabled_channels:
                self._send_push_notification(user_id, notification_type, data)

        def _create_in_app_notification(self, user_id: int,
                                       notification_type: NotificationType, data: dict):
            """Create in-app notification"""
            self.db.execute("""
                INSERT INTO notifications (user_id, type, data, is_read, created_at)
                VALUES (?, ?, ?, false, NOW())
            """, user_id, notification_type.value, json.dumps(data))

        def _send_email_notification(self, user_id: int,
                                     notification_type: NotificationType, data: dict):
            """Send email notification"""
            user = self.db.query("SELECT email, name FROM users WHERE user_id = ?", user_id)

            subject, body = self._format_email(notification_type, data)

            self.email_service.send(
                to=user.email,
                subject=subject,
                body=body
            )

        def _send_push_notification(self, user_id: int,
                                   notification_type: NotificationType, data: dict):
            """Send push notification"""
            devices = self.db.query("""
                SELECT device_token, platform
                FROM user_devices
                WHERE user_id = ? AND push_enabled = true
            """, user_id)

            title, body = self._format_push(notification_type, data)

            for device in devices:
                self.push_service.send(
                    device_token=device.device_token,
                    platform=device.platform,
                    title=title,
                    body=body,
                    data=data
                )

        def _format_email(self, notification_type: NotificationType,
                         data: dict) -> tuple:
            """Format email subject and body"""
            templates = {
                NotificationType.TASK_ASSIGNED: (
                    "New task assigned: {title}",
                    "You've been assigned to task '{title}' in project '{project}'"
                ),
                NotificationType.TASK_MENTIONED: (
                    "You were mentioned in a comment",
                    "{user} mentioned you in task '{title}'"
                )
            }

            subject_template, body_template = templates[notification_type]
            return (
                subject_template.format(**data),
                body_template.format(**data)
            )

        def batch_send_due_reminders(self):
            """Background job to send due date reminders"""
            # Find tasks due in 24 hours
            tasks = self.db.query("""
                SELECT task_id, title, assignee_id, project_id
                FROM tasks
                WHERE due_date BETWEEN NOW() AND NOW() + INTERVAL '24 hours'
                AND status != 'done'
            """)

            for task in tasks:
                self.send_notification(
                    user_id=task.assignee_id,
                    notification_type=NotificationType.TASK_DUE_SOON,
                    data={
                        "task_id": task.task_id,
                        "title": task.title,
                        "due_date": task.due_date
                    }
                )
    ```

    ---

    ### 5. Search and Filtering

    **Elasticsearch indexing and search:**

    ```python
    from elasticsearch import Elasticsearch

    class TaskSearchService:
        def __init__(self, es_client: Elasticsearch):
            self.es = es_client
            self.index = "tasks"

        def index_task(self, task: dict):
            """Index task for search"""
            doc = {
                "task_id": task["task_id"],
                "title": task["title"],
                "description": task["description"],
                "status": task["status"],
                "priority": task["priority"],
                "assignee_id": task["assignee_id"],
                "assignee_name": task["assignee_name"],
                "project_id": task["project_id"],
                "project_name": task["project_name"],
                "tags": task["tags"],
                "due_date": task["due_date"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            }

            self.es.index(index=self.index, id=task["task_id"], document=doc)

        def search_tasks(self, query: str, filters: dict, user_id: int,
                        sort: str = "relevance", limit: int = 20) -> dict:
            """Search tasks with filters"""
            # Get user's accessible projects
            accessible_projects = self._get_user_projects(user_id)

            # Build Elasticsearch query
            must_clauses = []
            filter_clauses = []

            # Text search
            if query:
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "description"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                })

            # Project filter (restrict to accessible projects)
            filter_clauses.append({
                "terms": {"project_id": accessible_projects}
            })

            # Additional filters
            if filters.get("project_id"):
                filter_clauses.append({"term": {"project_id": filters["project_id"]}})

            if filters.get("status"):
                filter_clauses.append({"term": {"status": filters["status"]}})

            if filters.get("assignee_id"):
                filter_clauses.append({"term": {"assignee_id": filters["assignee_id"]}})

            if filters.get("priority"):
                filter_clauses.append({"term": {"priority": filters["priority"]}})

            if filters.get("tags"):
                filter_clauses.append({"terms": {"tags": filters["tags"]}})

            if filters.get("due_date_before"):
                filter_clauses.append({
                    "range": {"due_date": {"lte": filters["due_date_before"]}}
                })

            # Sort
            sort_options = {
                "relevance": "_score",
                "due_date": {"due_date": {"order": "asc"}},
                "created_at": {"created_at": {"order": "desc"}},
                "updated_at": {"updated_at": {"order": "desc"}}
            }

            # Execute search
            response = self.es.search(
                index=self.index,
                query={
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}],
                        "filter": filter_clauses
                    }
                },
                sort=sort_options.get(sort, "_score"),
                size=limit,
                highlight={
                    "fields": {
                        "title": {},
                        "description": {}
                    }
                }
            )

            return {
                "tasks": [
                    {
                        **hit["_source"],
                        "score": hit["_score"],
                        "highlight": hit.get("highlight", {})
                    }
                    for hit in response["hits"]["hits"]
                ],
                "total": response["hits"]["total"]["value"]
            }
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Database** | 🟡 Maybe | Read replicas, connection pooling, query optimization |
    | **WebSocket Servers** | ✅ Yes | Horizontal scaling, 10K connections per server = 10 servers |
    | **Kafka** | ❌ No | Handles 1K+ events/sec easily |
    | **Elasticsearch** | 🟡 Maybe | Cluster with 3-5 nodes, sharding by project |
    | **Notification Workers** | 🟡 Maybe | Scale horizontally, batch email delivery |

    ---

    ## Performance Optimization

    ### 1. Database Query Optimization

    **Optimize task listing queries:**

    ```sql
    -- Use covering index for common queries
    CREATE INDEX idx_project_status_assignee
    ON tasks(project_id, status, assignee_id)
    INCLUDE (title, priority, due_date, created_at);

    -- Query uses index without table access
    SELECT task_id, title, priority, due_date, created_at
    FROM tasks
    WHERE project_id = 123 AND status = 'in_progress' AND assignee_id = 456;
    ```

    **Use materialized view for project stats:**

    ```sql
    CREATE MATERIALIZED VIEW project_task_stats AS
    SELECT
        project_id,
        COUNT(*) as total_tasks,
        COUNT(*) FILTER (WHERE status = 'done') as completed_tasks,
        COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress_tasks,
        COUNT(*) FILTER (WHERE due_date < NOW() AND status != 'done') as overdue_tasks
    FROM tasks
    GROUP BY project_id;

    -- Refresh periodically (every 5 minutes)
    REFRESH MATERIALIZED VIEW CONCURRENTLY project_task_stats;
    ```

    ---

    ### 2. Caching Strategy

    **Redis caching for hot data:**

    ```python
    class TaskCacheService:
        def __init__(self, redis_client, db):
            self.redis = redis_client
            self.db = db
            self.ttl = 300  # 5 minutes

        def get_task(self, task_id: int) -> dict:
            """Get task with caching"""
            cache_key = f"task:{task_id}"

            # Try cache first
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

            # Cache miss - fetch from DB
            task = self.db.query("SELECT * FROM tasks WHERE task_id = ?", task_id)

            # Store in cache
            self.redis.setex(cache_key, self.ttl, json.dumps(task))

            return task

        def invalidate_task(self, task_id: int):
            """Invalidate cache when task updated"""
            self.redis.delete(f"task:{task_id}")

        def get_project_tasks(self, project_id: int, status: str = None) -> List[dict]:
            """Get project tasks with caching"""
            cache_key = f"project:{project_id}:tasks:{status or 'all'}"

            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

            query = "SELECT * FROM tasks WHERE project_id = ?"
            params = [project_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            tasks = self.db.query(query, *params)

            # Cache for shorter TTL (project view changes frequently)
            self.redis.setex(cache_key, 60, json.dumps(tasks))

            return tasks
    ```

    ---

    ### 3. Batch Processing for Notifications

    **Batch email delivery to reduce overhead:**

    ```python
    class BatchNotificationWorker:
        def __init__(self, email_service):
            self.email_service = email_service
            self.batch = []
            self.batch_size = 100
            self.flush_interval = 5  # seconds

        def add_notification(self, user_id: int, notification: dict):
            """Add notification to batch"""
            self.batch.append({
                "user_id": user_id,
                "notification": notification
            })

            if len(self.batch) >= self.batch_size:
                self.flush()

        def flush(self):
            """Send all batched notifications"""
            if not self.batch:
                return

            # Group by user
            user_notifications = defaultdict(list)
            for item in self.batch:
                user_notifications[item["user_id"]].append(item["notification"])

            # Send digest email per user
            for user_id, notifications in user_notifications.items():
                user = self._get_user(user_id)

                if len(notifications) == 1:
                    # Single notification
                    self.email_service.send_single(user, notifications[0])
                else:
                    # Digest of multiple notifications
                    self.email_service.send_digest(user, notifications)

            self.batch.clear()

        async def run(self):
            """Background worker that flushes periodically"""
            while True:
                await asyncio.sleep(self.flush_interval)
                self.flush()
    ```

    ---

    ### 4. Database Partitioning

    **Partition activity log by time:**

    ```sql
    -- Partition by month for easier archival
    CREATE TABLE activities (
        activity_id BIGSERIAL,
        task_id BIGINT,
        actor_id BIGINT,
        action VARCHAR(50),
        changes JSONB,
        created_at TIMESTAMP NOT NULL
    ) PARTITION BY RANGE (created_at);

    -- Create partitions
    CREATE TABLE activities_2026_01 PARTITION OF activities
        FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

    CREATE TABLE activities_2026_02 PARTITION OF activities
        FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

    -- Auto-create future partitions via cron job
    ```

    ---

    ### 5. Monitoring and Metrics

    **Key metrics to track:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **API Latency (p95)** | < 200ms | > 500ms |
    | **Task Create Rate** | 100-200/sec | > 500/sec (unusual spike) |
    | **WebSocket Connections** | 100K | > 150K (need scaling) |
    | **Kafka Consumer Lag** | < 100 messages | > 1000 messages |
    | **Notification Delivery Rate** | > 99% | < 95% |
    | **Search Query Time** | < 300ms | > 1s |
    | **Database Connection Pool** | 70% utilization | > 90% |
    | **Cache Hit Rate** | > 80% | < 60% |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **PostgreSQL for primary storage** - ACID transactions, complex queries, foreign keys
    2. **DAG for task dependencies** - Prevent cycles with DFS cycle detection
    3. **State machine for workflows** - Enforce valid status transitions
    4. **WebSocket for real-time** - Instant collaboration with low latency
    5. **Kafka for events** - Reliable message delivery, replay capability
    6. **Elasticsearch for search** - Fast full-text search with filters
    7. **Cassandra for activity log** - High write throughput for audit trail
    8. **Redis for caching** - Hot task data, session management

    ---

    ## Interview Tips

    ✅ **Discuss task dependencies** - Explain DAG and cycle detection with DFS

    ✅ **Workflow automation** - State machines for status transitions, rule engine

    ✅ **Real-time updates** - WebSocket architecture, event broadcasting

    ✅ **Scalability** - Database sharding, caching strategy, read replicas

    ✅ **Notification delivery** - Multi-channel, batching, user preferences

    ✅ **Search functionality** - Elasticsearch indexing, filtering, relevance

    ---

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent circular dependencies?"** | Use DFS to detect cycles before adding edge, maintain DAG invariant |
    | **"How to handle task updates at scale?"** | Event-driven architecture, async workers, caching, read replicas |
    | **"What if WebSocket connection drops?"** | Client reconnects, fetch missed events via API using last_seen_event_id |
    | **"How to optimize search for large projects?"** | Elasticsearch sharding by project, index only active tasks, cache frequent queries |
    | **"How to ensure notification delivery?"** | Kafka for durability, retry with exponential backoff, dead letter queue for failures |
    | **"How to handle subtask completion propagation?"** | Background worker calculates parent progress: completed_subtasks / total_subtasks |
    | **"How to scale WebSocket servers?"** | Sticky sessions for load balancing, Redis pub/sub for cross-server messaging |
    | **"What if two users update task simultaneously?"** | Optimistic locking with version field, last-write-wins, or conflict resolution UI |

    ---

    ## Real-World Examples

    **Asana:**
    - Kanban boards, list views, timeline views
    - Task dependencies with critical path highlighting
    - Custom fields and templates
    - Rules for workflow automation

    **Jira:**
    - Issue types (Bug, Story, Epic)
    - Workflow customization per project
    - Advanced JQL search syntax
    - Sprint planning and velocity tracking

    **Trello:**
    - Card-based Kanban boards
    - Simple drag-and-drop interface
    - Power-ups for extensibility
    - Butler for automation

    **Linear:**
    - Fast keyboard-driven UI
    - Automatic issue assignment based on workload
    - Cycle-based planning
    - GitHub integration for automatic status updates

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Asana, Jira, Trello, Monday.com, ClickUp, Linear

---

*Master this problem and you'll understand task management systems, workflow automation, real-time collaboration, and dependency management - essential for building productivity tools.*
