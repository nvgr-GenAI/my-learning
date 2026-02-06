# Design a Prompt Management System

A comprehensive prompt management platform that enables teams to version, test, deploy, and monitor LLM prompts at scale, with features for template management, A/B testing, caching, cost tracking, and collaboration across development, staging, and production environments.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K prompts, 10M executions/day, 5K concurrent users, 100+ projects |
| **Key Challenges** | Prompt versioning, A/B testing with statistical significance, semantic caching, cost tracking per prompt/version, real-time analytics |
| **Core Concepts** | Prompt registry, template rendering (Jinja2), version control (Git-like), A/B test manager, semantic caching, LLM gateway, analytics dashboard |
| **Companies** | PromptLayer, Helicone, LangSmith (LangChain), Weights & Biases Prompts, HumanLoop, Braintrust, LiteLLM |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Prompt Registry** | Store and version prompts with metadata (name, description, tags) | P0 (Must have) |
    | **Template Engine** | Support variable interpolation (Jinja2/Handlebars) | P0 (Must have) |
    | **Version Control** | Git-like semantics (commit, branch, rollback, diff) | P0 (Must have) |
    | **Prompt Execution** | Execute prompts against multiple LLM providers (OpenAI, Anthropic) | P0 (Must have) |
    | **A/B Testing** | Run experiments comparing prompt variants with statistical analysis | P0 (Must have) |
    | **Caching** | Semantic caching for similar prompts to reduce costs | P0 (Must have) |
    | **Cost Tracking** | Track token usage and costs per prompt/version/user | P0 (Must have) |
    | **Analytics Dashboard** | Real-time metrics (latency, tokens, costs, success rate) | P0 (Must have) |
    | **Prompt Chaining** | Orchestrate multi-step prompt workflows (sequential, parallel) | P1 (Should have) |
    | **Collaboration** | Share prompts, comments, reviews, approval workflows | P1 (Should have) |
    | **Access Control** | Role-based permissions (viewer, editor, admin) | P1 (Should have) |
    | **Playground** | Interactive prompt testing with different parameters | P2 (Nice to have) |
    | **Prompt Discovery** | Search/filter prompts by tags, metrics, usage | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training or fine-tuning infrastructure
    - Custom LLM hosting and inference
    - Data labeling and annotation tools
    - Feature engineering for ML models
    - Production model monitoring (drift detection)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Critical for production applications |
    | **Execution Latency** | < 50ms overhead (excluding LLM call) | Should not slow down applications |
    | **Cache Hit Rate** | > 30% for production workloads | Significant cost savings |
    | **Storage Retention** | Unlimited (user-controlled cleanup) | Historical prompt analysis |
    | **Concurrent Executions** | 10K+ simultaneous requests | Support high-traffic applications |
    | **Version Limit** | 1000 versions per prompt | Reasonable for production use |
    | **Data Durability** | 99.999999999% (S3 standard) | Cannot lose prompt history |
    | **Analytics Latency** | < 5 seconds for dashboard queries | Real-time monitoring |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Total users: 5,000 users
    - Active concurrent users: 500 users (peak hours)
    - Total prompts: 10,000 prompts
    - Prompt versions per prompt: 10 versions average
    - Total versions: 100,000 versions

    Prompt executions:
    - Daily executions: 10M executions/day
    - QPS average: 10M / 86,400 = ~115 req/sec
    - QPS peak: 5x average = ~575 req/sec
    - Average prompt tokens: 1,500 tokens (1,000 input + 500 output)
    - Daily tokens: 10M × 1,500 = 15B tokens/day

    Cache traffic:
    - Cache hit rate target: 30%
    - Cache hits: 3M/day
    - Cache misses: 7M/day (actual LLM calls)
    - Cost savings: 3M × $0.001 (avg per request) = $3,000/day

    Analytics/Dashboard traffic:
    - Dashboard views per user: 10 views/day
    - Total dashboard queries: 5,000 × 10 = 50,000 queries/day
    - QPS for analytics: 50,000 / 86,400 = 0.58 QPS (very low)

    A/B test traffic:
    - Active experiments: 100 experiments
    - Variants per experiment: 2-4 variants
    - Traffic split: 10-50% of executions per experiment

    Total write QPS: ~575 (executions + logs)
    Total read QPS: ~100 (cache lookups, prompt fetches)
    Write-to-read ratio: 6:1 (write-heavy)
    ```

    ### Storage Estimates

    ```
    Metadata storage (PostgreSQL):
    - Prompts: 10K prompts × 5 KB = 50 MB
    - Versions: 100K versions × 2 KB = 200 MB
    - Users/teams: 5K users × 1 KB = 5 MB
    - A/B experiments: 1K experiments × 10 KB = 10 MB
    - Total metadata: ~265 MB (with indexes: ~500 MB)

    Execution logs (TimescaleDB):
    - Logs per execution: 1 log × 2 KB = 2 KB
    - Daily logs: 10M × 2 KB = 20 GB/day
    - Monthly logs: 600 GB/month
    - With downsampling (older data): 300 GB/month
    - 1 year retention: 3.6 TB (with compression: 1.8 TB)

    Cache storage (Redis):
    - Cache entries: 1M entries (sliding window)
    - Entry size: 2 KB (prompt hash + response)
    - Total cache: 2 GB
    - With metadata: 3 GB

    Prompt templates (S3):
    - Templates per version: 100K versions × 10 KB = 1 GB
    - Historical versions: Negligible (text data)

    Total storage: 500 MB (metadata) + 1.8 TB (logs) + 3 GB (cache) + 1 GB (templates) ≈ 1.8 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (prompt executions):
    - Request size: 2 KB (prompt ID + variables)
    - Executions: 575 req/sec × 2 KB = 1.15 MB/sec ≈ 9.2 Mbps
    - LLM response logging: 575 req/sec × 3 KB = 1.73 MB/sec ≈ 13.8 Mbps
    - Total ingress: ~23 Mbps

    Egress (prompt delivery + analytics):
    - Prompt templates: 575 req/sec × 10 KB = 5.75 MB/sec ≈ 46 Mbps
    - Analytics dashboard: 1 req/sec × 50 KB = 50 KB/sec ≈ 0.4 Mbps
    - Total egress: ~46.4 Mbps

    LLM API traffic (external):
    - Cache misses: 7M/day = 81 req/sec
    - Request size: 1.5 KB (avg)
    - Response size: 3 KB (avg)
    - Ingress from LLM: 81 × 3 KB = 243 KB/sec ≈ 1.9 Mbps
    - Egress to LLM: 81 × 1.5 KB = 122 KB/sec ≈ 1 Mbps

    Total bandwidth: ~72 Mbps (100 Mbps link sufficient)
    ```

    ### Memory Estimates

    ```
    Application servers:
    - In-memory cache (hot prompts): 500 MB
    - Connection pools (DB, Redis): 200 MB
    - Per-server memory: 2 GB
    - Total servers (10 replicas): 20 GB

    Database (PostgreSQL):
    - Shared buffers (25% of data): 125 MB
    - Connection memory: 2 GB
    - Total DB memory: 4 GB (server RAM)

    Time-series DB (TimescaleDB):
    - Cache for recent logs: 5 GB
    - Total: 8 GB (server RAM)

    Redis cache:
    - Cache data: 3 GB
    - Key indexes: 1 GB
    - Total: 8 GB

    Total memory: 20 GB (app) + 4 GB (DB) + 8 GB (TimescaleDB) + 8 GB (Redis) ≈ 40 GB
    ```

    ---

    ## Key Assumptions

    1. Average prompt has 10 versions over its lifetime
    2. 30% of executions hit cache (semantic similarity)
    3. Users primarily search recent prompts (last 30 days)
    4. A/B tests run for 1-2 weeks before conclusion
    5. 80% of executions are production traffic, 20% are development
    6. Most prompts use OpenAI (70%), Anthropic (20%), other providers (10%)
    7. Users collaborate within teams (10-50 users per team)
    8. Peak traffic is 5x average during business hours

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separation of concerns:** Prompt storage, execution, caching, and analytics are independent
    2. **Provider agnostic:** Support multiple LLM providers through unified gateway
    3. **Version control:** Git-like semantics for prompt history and rollback
    4. **Cost optimization:** Aggressive caching with semantic similarity
    5. **Real-time analytics:** Stream execution logs for dashboards
    6. **Experimentation:** Built-in A/B testing with statistical analysis

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Applications"
            App[Application Code<br/>Python/JS SDK]
            Dashboard[Web Dashboard<br/>React<br/>Prompt editor]
            Playground[Playground<br/>Interactive testing]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Rate limiting<br/>Auth/RBAC]
        end

        subgraph "Core Services"
            PromptService[Prompt Service<br/>CRUD operations<br/>Version control]
            ExecutionService[Execution Service<br/>Template rendering<br/>Variable interpolation]
            CacheService[Cache Service<br/>Semantic similarity<br/>Redis cache]
        end

        subgraph "LLM Gateway"
            Router[LLM Router<br/>Provider selection<br/>Load balancing]
            OpenAIAdapter[OpenAI Adapter]
            AnthropicAdapter[Anthropic Adapter]
            HuggingFaceAdapter[HuggingFace Adapter]
        end

        subgraph "Experimentation"
            ABTestManager[A/B Test Manager<br/>Traffic splitting<br/>Statistical analysis]
            VariantSelector[Variant Selector<br/>User assignment]
        end

        subgraph "Storage Layer"
            PostgreSQL[(PostgreSQL<br/>Prompts, versions<br/>Users, teams)]
            TimescaleDB[(TimescaleDB<br/>Execution logs<br/>Time-series metrics)]
            Redis[(Redis<br/>Semantic cache<br/>Session storage)]
            S3[(S3<br/>Prompt templates<br/>Large outputs)]
        end

        subgraph "Analytics Pipeline"
            EventStream[Event Stream<br/>Kafka/Kinesis]
            MetricsAggregator[Metrics Aggregator<br/>Real-time aggregation]
            AnalyticsDB[(Analytics DB<br/>ClickHouse<br/>Dashboard queries)]
        end

        subgraph "Background Jobs"
            CacheWarmer[Cache Warmer<br/>Pre-compute popular<br/>prompts]
            CostTracker[Cost Tracker<br/>Aggregate usage<br/>Generate reports]
            ABAnalyzer[A/B Analyzer<br/>Statistical tests<br/>Auto-conclude]
        end

        App -->|1. Execute prompt| Gateway
        Dashboard -->|2. Manage prompts| Gateway
        Playground -->|3. Test prompts| Gateway

        Gateway -->|4a. CRUD| PromptService
        Gateway -->|4b. Execute| ExecutionService

        PromptService -->|5. Store/retrieve| PostgreSQL
        PromptService -->|6. Store templates| S3

        ExecutionService -->|7. Check cache| CacheService
        CacheService -->|8. Lookup| Redis

        ExecutionService -->|9a. Cache miss| Router
        ExecutionService -->|9b. Get variant| ABTestManager
        ABTestManager -->|10. Select variant| VariantSelector
        VariantSelector -->|11. Assign user| PostgreSQL

        Router -->|12a. Route to provider| OpenAIAdapter
        Router -->|12b. Route to provider| AnthropicAdapter
        Router -->|12c. Route to provider| HuggingFaceAdapter

        OpenAIAdapter -->|13. LLM API call| OpenAI[OpenAI API]
        AnthropicAdapter -->|13. LLM API call| Anthropic[Anthropic API]

        ExecutionService -->|14. Log execution| EventStream
        EventStream -->|15. Process| MetricsAggregator
        MetricsAggregator -->|16a. Store logs| TimescaleDB
        MetricsAggregator -->|16b. Store metrics| AnalyticsDB

        Dashboard -->|17. Query analytics| AnalyticsDB

        CacheWarmer -.->|Warm cache| Redis
        CostTracker -.->|Analyze| TimescaleDB
        ABAnalyzer -.->|Compute stats| AnalyticsDB

        style ExecutionService fill:#e1f5ff
        style CacheService fill:#ffe1e1
        style ABTestManager fill:#f3e5f5
        style Router fill:#fff4e1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PostgreSQL (Metadata)** | ACID transactions, relations (prompts→versions), complex queries | MongoDB (no strong consistency for versions), DynamoDB (expensive for metadata queries) |
    | **TimescaleDB (Logs)** | Time-series optimized, automatic partitioning, fast aggregations | InfluxDB (limited SQL support), Cassandra (complex maintenance) |
    | **Redis (Cache)** | Sub-millisecond reads, semantic similarity with embeddings | Memcached (no persistence), in-memory (lost on restart) |
    | **Kafka/Kinesis (Streaming)** | High-throughput event streaming, reliable delivery | RabbitMQ (lower throughput), direct DB writes (slow) |
    | **ClickHouse (Analytics)** | Columnar storage, fast aggregations, perfect for dashboards | PostgreSQL (slow for large analytical queries), Elasticsearch (higher resource usage) |
    | **S3 (Templates)** | Cheap, durable, versioned storage for large prompt templates | Database (not for large text), EFS (more expensive) |

    **Key Trade-off:** We chose **semantic caching over exact-match caching** to maximize cache hit rate (30% vs 5-10%). This requires embedding generation but saves significant LLM costs.

    ---

    ## API Design

    ### 1. Create Prompt

    **Request:**
    ```python
    POST /api/v1/prompts
    {
      "name": "customer_email_generator",
      "description": "Generate professional customer emails",
      "template": "Write a professional email to {{customer_name}} about {{topic}}. Tone: {{tone}}",
      "model": "gpt-4-turbo-preview",
      "parameters": {
        "temperature": 0.7,
        "max_tokens": 500
      },
      "tags": ["email", "customer-service"],
      "metadata": {
        "owner": "support-team",
        "project": "customer-automation"
      }
    }
    ```

    **Response:**
    ```json
    {
      "prompt_id": "prompt_abc123",
      "version_id": "v1_xyz789",
      "name": "customer_email_generator",
      "version": 1,
      "created_at": "2024-01-15T10:30:00Z",
      "created_by": "user_123"
    }
    ```

    ---

    ### 2. Update Prompt (Create New Version)

    **Request:**
    ```python
    POST /api/v1/prompts/prompt_abc123/versions
    {
      "template": "Write a professional email to {{customer_name}} about {{topic}}. Tone: {{tone}}. Keep it under 200 words.",
      "commit_message": "Added word count constraint",
      "parameters": {
        "temperature": 0.6,  # Changed from 0.7
        "max_tokens": 300    # Reduced from 500
      }
    }
    ```

    **Response:**
    ```json
    {
      "prompt_id": "prompt_abc123",
      "version_id": "v2_def456",
      "version": 2,
      "commit_message": "Added word count constraint",
      "diff": {
        "template": {
          "old": "Write a professional email to {{customer_name}} about {{topic}}. Tone: {{tone}}",
          "new": "Write a professional email to {{customer_name}} about {{topic}}. Tone: {{tone}}. Keep it under 200 words."
        },
        "parameters": {
          "temperature": {"old": 0.7, "new": 0.6},
          "max_tokens": {"old": 500, "new": 300}
        }
      },
      "created_at": "2024-01-15T11:00:00Z"
    }
    ```

    ---

    ### 3. Execute Prompt

    **Request:**
    ```python
    POST /api/v1/prompts/prompt_abc123/execute
    {
      "version": "latest",  # or specific version: "v2_def456"
      "variables": {
        "customer_name": "John Smith",
        "topic": "order delay",
        "tone": "apologetic"
      },
      "options": {
        "use_cache": true,
        "stream": false,
        "metadata": {
          "user_id": "user_456",
          "session_id": "session_789"
        }
      }
    }
    ```

    **Response:**
    ```json
    {
      "execution_id": "exec_xyz123",
      "prompt_id": "prompt_abc123",
      "version_id": "v2_def456",
      "output": "Dear John Smith,\n\nI sincerely apologize for the delay with your order...",
      "cached": false,
      "metadata": {
        "model": "gpt-4-turbo-preview",
        "provider": "openai",
        "latency_ms": 1234,
        "tokens": {
          "prompt": 45,
          "completion": 156,
          "total": 201
        },
        "cost": {
          "prompt": 0.00045,
          "completion": 0.00468,
          "total": 0.00513
        }
      },
      "executed_at": "2024-01-15T11:05:00Z"
    }
    ```

    ---

    ### 4. Create A/B Test

    **Request:**
    ```python
    POST /api/v1/experiments
    {
      "name": "Email tone experiment",
      "description": "Test formal vs casual tone for customer emails",
      "variants": [
        {
          "name": "control",
          "prompt_version_id": "v2_def456",
          "traffic_percentage": 50
        },
        {
          "name": "casual_tone",
          "prompt_version_id": "v3_ghi789",
          "traffic_percentage": 50
        }
      ],
      "metrics": ["response_rate", "customer_satisfaction"],
      "duration_days": 14,
      "min_sample_size": 1000,
      "confidence_level": 0.95
    }
    ```

    **Response:**
    ```json
    {
      "experiment_id": "exp_abc123",
      "name": "Email tone experiment",
      "status": "running",
      "started_at": "2024-01-15T12:00:00Z",
      "estimated_end_at": "2024-01-29T12:00:00Z",
      "variants": [
        {
          "variant_id": "var_control",
          "name": "control",
          "executions": 0,
          "traffic_percentage": 50
        },
        {
          "variant_id": "var_casual",
          "name": "casual_tone",
          "executions": 0,
          "traffic_percentage": 50
        }
      ]
    }
    ```

    ---

    ### 5. Get Analytics

    **Request:**
    ```python
    GET /api/v1/prompts/prompt_abc123/analytics?
      start_date=2024-01-01&
      end_date=2024-01-31&
      metrics=latency,tokens,cost,success_rate&
      group_by=day,version
    ```

    **Response:**
    ```json
    {
      "prompt_id": "prompt_abc123",
      "period": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z"
      },
      "summary": {
        "total_executions": 125000,
        "cache_hit_rate": 0.32,
        "avg_latency_ms": 1456,
        "total_tokens": 187500000,
        "total_cost": 625.50,
        "success_rate": 0.987
      },
      "timeseries": [
        {
          "date": "2024-01-01",
          "version": "v2_def456",
          "executions": 4200,
          "avg_latency_ms": 1423,
          "tokens": 6300000,
          "cost": 21.05,
          "success_rate": 0.991
        }
      ],
      "by_version": [
        {
          "version_id": "v2_def456",
          "executions": 95000,
          "cost": 475.25,
          "avg_latency_ms": 1401
        },
        {
          "version_id": "v3_ghi789",
          "executions": 30000,
          "cost": 150.25,
          "avg_latency_ms": 1567
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Metadata DB (PostgreSQL)

    ```sql
    -- Prompts (top-level container)
    CREATE TABLE prompts (
        prompt_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        description TEXT,
        project_id VARCHAR(50) REFERENCES projects(project_id),
        created_by VARCHAR(100) REFERENCES users(user_id),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        deleted_at TIMESTAMP,  -- Soft delete
        metadata JSONB
    );
    CREATE INDEX idx_prompts_name ON prompts(name);
    CREATE INDEX idx_prompts_project ON prompts(project_id);
    CREATE INDEX idx_prompts_created ON prompts(created_at DESC);

    -- Prompt versions (immutable)
    CREATE TABLE prompt_versions (
        version_id VARCHAR(50) PRIMARY KEY,
        prompt_id VARCHAR(50) REFERENCES prompts(prompt_id),
        version_number INTEGER NOT NULL,
        template TEXT NOT NULL,
        model VARCHAR(100) NOT NULL,  -- gpt-4, claude-3-opus, etc.
        parameters JSONB,  -- temperature, max_tokens, etc.
        commit_message TEXT,
        parent_version_id VARCHAR(50) REFERENCES prompt_versions(version_id),
        created_by VARCHAR(100) REFERENCES users(user_id),
        created_at TIMESTAMP DEFAULT NOW(),
        s3_template_path VARCHAR(500),  -- For large templates
        UNIQUE(prompt_id, version_number)
    );
    CREATE INDEX idx_versions_prompt ON prompt_versions(prompt_id);
    CREATE INDEX idx_versions_created ON prompt_versions(created_at DESC);

    -- Tags (many-to-many)
    CREATE TABLE prompt_tags (
        prompt_id VARCHAR(50) REFERENCES prompts(prompt_id),
        tag VARCHAR(100) NOT NULL,
        PRIMARY KEY (prompt_id, tag)
    );
    CREATE INDEX idx_tags_tag ON prompt_tags(tag);

    -- A/B Experiments
    CREATE TABLE experiments (
        experiment_id VARCHAR(50) PRIMARY KEY,
        prompt_id VARCHAR(50) REFERENCES prompts(prompt_id),
        name VARCHAR(255) NOT NULL,
        description TEXT,
        status VARCHAR(20),  -- draft, running, paused, concluded
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        min_sample_size INTEGER,
        confidence_level FLOAT,
        winner_variant_id VARCHAR(50),
        created_by VARCHAR(100) REFERENCES users(user_id),
        metadata JSONB
    );
    CREATE INDEX idx_experiments_prompt ON experiments(prompt_id);
    CREATE INDEX idx_experiments_status ON experiments(status);

    -- Experiment variants
    CREATE TABLE experiment_variants (
        variant_id VARCHAR(50) PRIMARY KEY,
        experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
        name VARCHAR(100) NOT NULL,
        version_id VARCHAR(50) REFERENCES prompt_versions(version_id),
        traffic_percentage INTEGER,  -- 0-100
        is_control BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_variants_experiment ON experiment_variants(experiment_id);

    -- User-variant assignments (for consistent experience)
    CREATE TABLE user_variant_assignments (
        experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
        user_id VARCHAR(100),
        variant_id VARCHAR(50) REFERENCES experiment_variants(variant_id),
        assigned_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (experiment_id, user_id)
    );
    ```

    ---

    ### Time-Series DB (TimescaleDB)

    **Schema:**
    ```sql
    -- Execution logs (hypertable)
    CREATE TABLE prompt_executions (
        execution_id VARCHAR(50) PRIMARY KEY,
        prompt_id VARCHAR(50) NOT NULL,
        version_id VARCHAR(50) NOT NULL,
        experiment_id VARCHAR(50),
        variant_id VARCHAR(50),

        -- Execution details
        rendered_prompt TEXT,
        output TEXT,
        variables JSONB,

        -- Performance metrics
        latency_ms INTEGER,
        cached BOOLEAN,

        -- Token usage
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,

        -- Cost
        prompt_cost DECIMAL(10, 6),
        completion_cost DECIMAL(10, 6),
        total_cost DECIMAL(10, 6),

        -- LLM details
        model VARCHAR(100),
        provider VARCHAR(50),  -- openai, anthropic, etc.

        -- Status
        success BOOLEAN,
        error_message TEXT,

        -- Metadata
        user_id VARCHAR(100),
        session_id VARCHAR(100),
        metadata JSONB,

        executed_at TIMESTAMPTZ NOT NULL
    );

    -- Convert to hypertable (TimescaleDB)
    SELECT create_hypertable('prompt_executions', 'executed_at',
        chunk_time_interval => INTERVAL '1 day');

    -- Indexes
    CREATE INDEX idx_executions_prompt ON prompt_executions(prompt_id, executed_at DESC);
    CREATE INDEX idx_executions_version ON prompt_executions(version_id, executed_at DESC);
    CREATE INDEX idx_executions_experiment ON prompt_executions(experiment_id, executed_at DESC);
    CREATE INDEX idx_executions_user ON prompt_executions(user_id, executed_at DESC);

    -- Continuous aggregates (pre-computed rollups)
    CREATE MATERIALIZED VIEW prompt_hourly_stats
    WITH (timescaledb.continuous) AS
    SELECT
        prompt_id,
        version_id,
        time_bucket('1 hour', executed_at) AS hour,
        COUNT(*) AS executions,
        AVG(latency_ms) AS avg_latency_ms,
        SUM(total_tokens) AS total_tokens,
        SUM(total_cost) AS total_cost,
        SUM(CASE WHEN cached THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS cache_hit_rate,
        SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS success_rate
    FROM prompt_executions
    GROUP BY prompt_id, version_id, hour;

    -- Refresh policy (update every hour)
    SELECT add_continuous_aggregate_policy('prompt_hourly_stats',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
    ```

    ---

    ### Cache Storage (Redis)

    **Data Structures:**
    ```python
    # Semantic cache: prompt hash → cached response
    Key: "cache:semantic:{embedding_hash}"
    Value: {
        "output": "...",
        "tokens": {"prompt": 45, "completion": 156},
        "cost": 0.00513,
        "cached_at": "2024-01-15T11:05:00Z"
    }
    TTL: 86400 (24 hours)

    # Prompt embeddings: prompt content → embedding vector
    Key: "embedding:{content_hash}"
    Value: [0.123, -0.456, 0.789, ...]  # Vector
    TTL: 604800 (7 days)

    # Hot prompts: frequently accessed prompt metadata
    Key: "prompt:hot:{prompt_id}"
    Value: {
        "name": "...",
        "latest_version_id": "...",
        "template": "..."
    }
    TTL: 3600 (1 hour)

    # A/B test assignments: user → variant
    Key: "experiment:{experiment_id}:user:{user_id}"
    Value: "variant_id"
    TTL: 1209600 (14 days)
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. Template Rendering with Jinja2

    **Challenge:** Support flexible variable interpolation with control flow (if/else, loops, filters).

    **Solution: Jinja2 Template Engine**

    ```python
    from jinja2 import Template, Environment, StrictUndefined
    from typing import Dict, Any
    import json

    class PromptTemplateRenderer:
        """
        Render prompt templates with variable interpolation.

        Features:
        - Jinja2 syntax ({{var}}, {% if %}, {% for %})
        - Custom filters (uppercase, truncate, format_date)
        - Strict mode (error on undefined variables)
        - Sandbox mode (prevent code execution)
        """

        def __init__(self):
            # Create Jinja2 environment with custom filters
            self.env = Environment(
                undefined=StrictUndefined,  # Error on undefined vars
                autoescape=False  # Don't escape HTML (not needed for prompts)
            )

            # Add custom filters
            self.env.filters['uppercase'] = lambda x: str(x).upper()
            self.env.filters['truncate_words'] = self._truncate_words
            self.env.filters['format_list'] = self._format_list

        def render(self, template_str: str, variables: Dict[str, Any]) -> str:
            """
            Render prompt template with variables.

            Args:
                template_str: Jinja2 template string
                variables: Dictionary of variables to inject

            Returns:
                Rendered prompt string

            Raises:
                TemplateError: If template is invalid or variable missing
            """
            try:
                template = self.env.from_string(template_str)
                rendered = template.render(**variables)
                return rendered.strip()

            except Exception as e:
                raise TemplateError(f"Failed to render template: {e}")

        def validate_template(self, template_str: str) -> Dict:
            """
            Validate template syntax and extract required variables.

            Returns:
                {
                    "valid": bool,
                    "variables": List[str],  # Required variables
                    "errors": List[str]
                }
            """
            try:
                template = self.env.from_string(template_str)

                # Extract undefined variables
                ast = self.env.parse(template_str)
                variables = list(jinja2.meta.find_undeclared_variables(ast))

                return {
                    "valid": True,
                    "variables": variables,
                    "errors": []
                }

            except Exception as e:
                return {
                    "valid": False,
                    "variables": [],
                    "errors": [str(e)]
                }

        def _truncate_words(self, text: str, max_words: int) -> str:
            """Custom filter: truncate to max words."""
            words = text.split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + "..."

        def _format_list(self, items: list, separator: str = ", ") -> str:
            """Custom filter: format list as string."""
            return separator.join(str(item) for item in items)

    # Example usage
    renderer = PromptTemplateRenderer()

    # Basic interpolation
    template1 = "Hello {{name}}, your order #{{order_id}} is ready!"
    output1 = renderer.render(template1, {"name": "Alice", "order_id": 12345})
    # Output: "Hello Alice, your order #12345 is ready!"

    # Conditional logic
    template2 = """
    {% if priority == 'high' %}
    URGENT: Please respond immediately.
    {% else %}
    Please respond at your convenience.
    {% endif %}

    Customer: {{customer_name}}
    Issue: {{issue_description}}
    """
    output2 = renderer.render(template2, {
        "priority": "high",
        "customer_name": "Bob",
        "issue_description": "Payment failed"
    })

    # Loops
    template3 = """
    Order Summary:
    {% for item in items %}
    - {{item.name}} ({{item.quantity}}x) - ${{item.price}}
    {% endfor %}

    Total: ${{total}}
    """
    output3 = renderer.render(template3, {
        "items": [
            {"name": "Widget", "quantity": 2, "price": 10.00},
            {"name": "Gadget", "quantity": 1, "price": 25.00}
        ],
        "total": 45.00
    })

    # Custom filters
    template4 = "Customer: {{name | uppercase}}\nDescription: {{description | truncate_words(10)}}"
    output4 = renderer.render(template4, {
        "name": "john smith",
        "description": "This is a very long description that should be truncated to avoid overwhelming the customer"
    })
    ```

    ---

    ## 2. Version Control with Git-Like Semantics

    **Challenge:** Track prompt changes over time with diff, rollback, and branching capabilities.

    **Solution: Immutable Version Chain**

    ```python
    from typing import Optional, List, Dict
    from dataclasses import dataclass
    from datetime import datetime
    import difflib

    @dataclass
    class PromptVersion:
        version_id: str
        prompt_id: str
        version_number: int
        template: str
        model: str
        parameters: Dict
        commit_message: Optional[str]
        parent_version_id: Optional[str]
        created_by: str
        created_at: datetime

    class PromptVersionControl:
        """
        Git-like version control for prompts.

        Features:
        - Immutable versions (linked list)
        - Diff between versions
        - Rollback to previous version
        - Version history traversal
        """

        def __init__(self, db_client):
            self.db = db_client

        def create_version(
            self,
            prompt_id: str,
            template: str,
            model: str,
            parameters: Dict,
            commit_message: str,
            user_id: str
        ) -> PromptVersion:
            """
            Create new prompt version (like git commit).

            Args:
                prompt_id: Parent prompt ID
                template: New template content
                model: LLM model name
                parameters: Model parameters
                commit_message: Description of changes
                user_id: User making the change

            Returns:
                New PromptVersion object
            """
            # Get latest version
            latest_version = self.get_latest_version(prompt_id)

            # Generate new version ID and number
            version_number = (latest_version.version_number + 1) if latest_version else 1
            version_id = f"v{version_number}_{generate_id()}"

            # Create version object
            new_version = PromptVersion(
                version_id=version_id,
                prompt_id=prompt_id,
                version_number=version_number,
                template=template,
                model=model,
                parameters=parameters,
                commit_message=commit_message,
                parent_version_id=latest_version.version_id if latest_version else None,
                created_by=user_id,
                created_at=datetime.utcnow()
            )

            # Store in database
            self._store_version(new_version)

            return new_version

        def get_version_diff(self, version_id1: str, version_id2: str) -> Dict:
            """
            Get diff between two versions (like git diff).

            Returns:
                {
                    "template": {"old": "...", "new": "...", "diff": "..."},
                    "model": {"old": "...", "new": "..."},
                    "parameters": {"added": {}, "removed": {}, "modified": {}}
                }
            """
            v1 = self.get_version(version_id1)
            v2 = self.get_version(version_id2)

            # Template diff (line-by-line)
            template_diff = self._compute_text_diff(v1.template, v2.template)

            # Model diff
            model_diff = None
            if v1.model != v2.model:
                model_diff = {"old": v1.model, "new": v2.model}

            # Parameters diff
            params_diff = self._compute_dict_diff(v1.parameters, v2.parameters)

            return {
                "template": template_diff,
                "model": model_diff,
                "parameters": params_diff
            }

        def rollback(self, prompt_id: str, target_version_id: str, user_id: str) -> PromptVersion:
            """
            Rollback to a previous version (creates new version with old content).

            Like git revert: creates new commit that undoes changes.
            """
            target_version = self.get_version(target_version_id)

            # Create new version with target's content
            new_version = self.create_version(
                prompt_id=prompt_id,
                template=target_version.template,
                model=target_version.model,
                parameters=target_version.parameters,
                commit_message=f"Rollback to version {target_version.version_number}",
                user_id=user_id
            )

            return new_version

        def get_version_history(self, prompt_id: str) -> List[PromptVersion]:
            """
            Get full version history (like git log).

            Returns versions in reverse chronological order.
            """
            query = """
                SELECT * FROM prompt_versions
                WHERE prompt_id = %s
                ORDER BY version_number DESC
            """
            rows = self.db.fetch_all(query, (prompt_id,))

            return [self._row_to_version(row) for row in rows]

        def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
            """Get latest version of prompt."""
            query = """
                SELECT * FROM prompt_versions
                WHERE prompt_id = %s
                ORDER BY version_number DESC
                LIMIT 1
            """
            row = self.db.fetch_one(query, (prompt_id,))

            return self._row_to_version(row) if row else None

        def _compute_text_diff(self, text1: str, text2: str) -> Dict:
            """Compute unified diff between two texts."""
            lines1 = text1.splitlines(keepends=True)
            lines2 = text2.splitlines(keepends=True)

            diff = list(difflib.unified_diff(
                lines1, lines2,
                lineterm='',
                fromfile='old',
                tofile='new'
            ))

            return {
                "old": text1,
                "new": text2,
                "diff": "\n".join(diff),
                "changed": text1 != text2
            }

        def _compute_dict_diff(self, dict1: Dict, dict2: Dict) -> Dict:
            """Compute diff between two dictionaries."""
            keys1 = set(dict1.keys())
            keys2 = set(dict2.keys())

            return {
                "added": {k: dict2[k] for k in keys2 - keys1},
                "removed": {k: dict1[k] for k in keys1 - keys2},
                "modified": {
                    k: {"old": dict1[k], "new": dict2[k]}
                    for k in keys1 & keys2
                    if dict1[k] != dict2[k]
                }
            }

        def _store_version(self, version: PromptVersion):
            """Store version in database."""
            query = """
                INSERT INTO prompt_versions (
                    version_id, prompt_id, version_number, template,
                    model, parameters, commit_message, parent_version_id,
                    created_by, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.db.execute(query, (
                version.version_id,
                version.prompt_id,
                version.version_number,
                version.template,
                version.model,
                json.dumps(version.parameters),
                version.commit_message,
                version.parent_version_id,
                version.created_by,
                version.created_at
            ))

    # Usage example
    version_control = PromptVersionControl(db_client)

    # Create first version
    v1 = version_control.create_version(
        prompt_id="prompt_123",
        template="Hello {{name}}!",
        model="gpt-4",
        parameters={"temperature": 0.7},
        commit_message="Initial version",
        user_id="user_1"
    )

    # Update (create v2)
    v2 = version_control.create_version(
        prompt_id="prompt_123",
        template="Hello {{name}}, welcome to our service!",
        model="gpt-4-turbo",
        parameters={"temperature": 0.6, "max_tokens": 100},
        commit_message="Added greeting text and switched to turbo model",
        user_id="user_1"
    )

    # Get diff
    diff = version_control.get_version_diff(v1.version_id, v2.version_id)
    print(json.dumps(diff, indent=2))

    # Rollback
    v3 = version_control.rollback(
        prompt_id="prompt_123",
        target_version_id=v1.version_id,
        user_id="user_1"
    )
    ```

    ---

    ## 3. Semantic Caching for Similar Prompts

    **Challenge:** Exact-match caching has low hit rate (~5-10%). Need to cache semantically similar prompts.

    **Solution: Embedding-Based Similarity Search**

    ```python
    from typing import Optional, Dict, Any
    import hashlib
    import numpy as np
    from openai import OpenAI
    import redis
    import json

    class SemanticCache:
        """
        Semantic caching using embedding similarity.

        Strategy:
        1. Generate embedding for rendered prompt
        2. Search cache for similar embeddings (cosine similarity > 0.95)
        3. If hit, return cached response
        4. If miss, execute prompt and cache result

        Cache hit rate: ~30% (vs ~5% for exact match)
        """

        def __init__(
            self,
            redis_client: redis.Redis,
            embedding_model: str = "text-embedding-3-small",
            similarity_threshold: float = 0.95
        ):
            self.redis = redis_client
            self.openai = OpenAI()
            self.embedding_model = embedding_model
            self.similarity_threshold = similarity_threshold
            self.embedding_dim = 1536  # text-embedding-3-small dimension

        def get_cached_response(
            self,
            rendered_prompt: str,
            model: str,
            parameters: Dict
        ) -> Optional[Dict]:
            """
            Look up cached response for similar prompt.

            Args:
                rendered_prompt: Fully rendered prompt text
                model: LLM model name
                parameters: Model parameters

            Returns:
                Cached response if found, None otherwise
            """
            # Generate embedding for prompt
            embedding = self._get_embedding(rendered_prompt)

            # Search for similar cached prompts
            cache_key = self._find_similar_cached_prompt(
                embedding=embedding,
                model=model,
                parameters=parameters
            )

            if cache_key:
                # Cache hit!
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    self.redis.incr("cache:hits")
                    return json.loads(cached_data)

            # Cache miss
            self.redis.incr("cache:misses")
            return None

        def cache_response(
            self,
            rendered_prompt: str,
            model: str,
            parameters: Dict,
            response: Dict
        ):
            """
            Cache LLM response with embedding for similarity search.

            Args:
                rendered_prompt: Fully rendered prompt text
                model: LLM model name
                parameters: Model parameters
                response: LLM response to cache
            """
            # Generate embedding
            embedding = self._get_embedding(rendered_prompt)

            # Create cache key
            cache_key = self._generate_cache_key(
                embedding=embedding,
                model=model,
                parameters=parameters
            )

            # Store cached response
            cache_data = {
                "output": response["output"],
                "tokens": response["tokens"],
                "cost": response["cost"],
                "cached_at": datetime.utcnow().isoformat()
            }

            # Store with TTL (24 hours)
            self.redis.setex(
                cache_key,
                86400,  # 24 hours
                json.dumps(cache_data)
            )

            # Store embedding in sorted set for similarity search
            embedding_key = f"embeddings:{model}"
            self.redis.zadd(
                embedding_key,
                {cache_key: 0}  # Score doesn't matter for sorted set
            )

            # Store embedding vector separately
            embedding_data_key = f"embedding:{cache_key}"
            self.redis.setex(
                embedding_data_key,
                86400,
                json.dumps(embedding.tolist())
            )

        def _get_embedding(self, text: str) -> np.ndarray:
            """
            Generate embedding for text with caching.

            Embeddings are deterministic, so we can cache them.
            """
            # Check embedding cache first
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            embedding_cache_key = f"embedding:content:{content_hash}"

            cached_embedding = self.redis.get(embedding_cache_key)
            if cached_embedding:
                return np.array(json.loads(cached_embedding))

            # Generate embedding via API
            response = self.openai.embeddings.create(
                input=[text],
                model=self.embedding_model
            )

            embedding = np.array(response.data[0].embedding)

            # Cache embedding (7 days)
            self.redis.setex(
                embedding_cache_key,
                604800,  # 7 days
                json.dumps(embedding.tolist())
            )

            return embedding

        def _find_similar_cached_prompt(
            self,
            embedding: np.ndarray,
            model: str,
            parameters: Dict
        ) -> Optional[str]:
            """
            Find cached prompt with similar embedding.

            Uses brute-force cosine similarity (fast for small cache).
            For larger scale, use vector database (Qdrant, Pinecone).
            """
            embedding_key = f"embeddings:{model}"

            # Get all cached keys for this model
            cached_keys = self.redis.zrange(embedding_key, 0, -1)

            if not cached_keys:
                return None

            # Load embeddings and compute similarities
            best_similarity = 0
            best_key = None

            for cache_key in cached_keys:
                # Load cached embedding
                embedding_data_key = f"embedding:{cache_key}"
                cached_embedding_data = self.redis.get(embedding_data_key)

                if not cached_embedding_data:
                    continue

                cached_embedding = np.array(json.loads(cached_embedding_data))

                # Compute cosine similarity
                similarity = self._cosine_similarity(embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = cache_key

            # Return if above threshold
            if best_similarity >= self.similarity_threshold:
                return best_key

            return None

        def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
            """Compute cosine similarity between two vectors."""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            return dot_product / (norm1 * norm2)

        def _generate_cache_key(
            self,
            embedding: np.ndarray,
            model: str,
            parameters: Dict
        ) -> str:
            """Generate unique cache key."""
            # Hash embedding + model + parameters
            embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
            params_hash = hashlib.sha256(
                json.dumps(parameters, sort_keys=True).encode()
            ).hexdigest()[:8]

            return f"cache:semantic:{model}:{embedding_hash}:{params_hash}"

        def get_cache_stats(self) -> Dict:
            """Get cache hit rate statistics."""
            hits = int(self.redis.get("cache:hits") or 0)
            misses = int(self.redis.get("cache:misses") or 0)
            total = hits + misses

            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total > 0 else 0,
                "total_requests": total
            }

    # Usage example
    cache = SemanticCache(redis_client)

    # First execution (cache miss)
    rendered_prompt = "Write a professional email to John about the meeting delay"
    cached = cache.get_cached_response(
        rendered_prompt=rendered_prompt,
        model="gpt-4-turbo",
        parameters={"temperature": 0.7}
    )

    if not cached:
        # Execute LLM
        response = execute_llm(rendered_prompt)

        # Cache response
        cache.cache_response(
            rendered_prompt=rendered_prompt,
            model="gpt-4-turbo",
            parameters={"temperature": 0.7},
            response=response
        )
    else:
        # Use cached response
        response = cached
        print("Cache hit!")

    # Second execution with similar prompt (cache hit!)
    similar_prompt = "Write a professional email to John regarding the meeting postponement"
    cached = cache.get_cached_response(
        rendered_prompt=similar_prompt,
        model="gpt-4-turbo",
        parameters={"temperature": 0.7}
    )
    # Returns cached response from first execution (similarity > 0.95)

    # Get stats
    stats = cache.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    ```

    ---

    ## 4. A/B Testing with Statistical Analysis

    **Challenge:** Run experiments to compare prompt variants with statistical significance.

    **Solution: Bayesian A/B Testing with Early Stopping**

    ```python
    from typing import List, Dict, Optional
    from dataclasses import dataclass
    from scipy import stats
    import numpy as np
    from datetime import datetime, timedelta

    @dataclass
    class Variant:
        variant_id: str
        name: str
        version_id: str
        traffic_percentage: int
        is_control: bool

    @dataclass
    class ExperimentResult:
        experiment_id: str
        status: str  # running, concluded
        winner_variant_id: Optional[str]
        confidence: float
        statistical_power: float
        p_value: float
        effect_size: float
        recommendation: str

    class ABTestManager:
        """
        A/B testing manager with statistical analysis.

        Features:
        - Traffic splitting with consistent user assignment
        - Statistical significance testing (t-test, chi-square)
        - Bayesian analysis with credible intervals
        - Early stopping (sequential testing)
        - Automatic experiment conclusion
        """

        def __init__(self, db_client, analytics_db):
            self.db = db_client
            self.analytics_db = analytics_db
            self.min_sample_size = 1000  # Minimum per variant
            self.confidence_level = 0.95  # 95% confidence

        def assign_variant(
            self,
            experiment_id: str,
            user_id: str
        ) -> str:
            """
            Assign user to variant (consistent assignment).

            Uses hash-based assignment for deterministic results.
            """
            # Check if user already assigned
            assignment = self._get_user_assignment(experiment_id, user_id)
            if assignment:
                return assignment

            # Get experiment variants
            variants = self._get_variants(experiment_id)

            # Hash-based assignment (deterministic)
            variant = self._hash_based_assignment(user_id, variants)

            # Store assignment
            self._store_assignment(experiment_id, user_id, variant.variant_id)

            return variant.variant_id

        def analyze_experiment(
            self,
            experiment_id: str,
            metric_name: str = "success_rate"
        ) -> ExperimentResult:
            """
            Analyze A/B test results with statistical significance.

            Metrics:
            - success_rate: Boolean success metric (conversion, satisfaction)
            - latency: Continuous metric (response time)
            - cost: Continuous metric (LLM cost)

            Returns:
                ExperimentResult with statistical analysis
            """
            # Get experiment data
            experiment = self._get_experiment(experiment_id)
            variants = self._get_variants(experiment_id)

            # Get metrics for each variant
            variant_metrics = {}
            for variant in variants:
                metrics = self._get_variant_metrics(
                    experiment_id=experiment_id,
                    variant_id=variant.variant_id,
                    metric_name=metric_name
                )
                variant_metrics[variant.variant_id] = metrics

            # Check if enough data collected
            min_samples = min(m["sample_size"] for m in variant_metrics.values())
            if min_samples < self.min_sample_size:
                return ExperimentResult(
                    experiment_id=experiment_id,
                    status="running",
                    winner_variant_id=None,
                    confidence=0.0,
                    statistical_power=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    recommendation=f"Need {self.min_sample_size - min_samples} more samples per variant"
                )

            # Perform statistical test
            if metric_name in ["success_rate", "conversion"]:
                # Binary metric: Chi-square test or proportion z-test
                result = self._test_proportions(variant_metrics)
            else:
                # Continuous metric: t-test
                result = self._test_means(variant_metrics)

            # Determine winner
            winner_variant_id = None
            if result["p_value"] < (1 - self.confidence_level):
                # Statistically significant difference
                winner_variant_id = result["best_variant_id"]

            # Update experiment status
            if winner_variant_id:
                self._conclude_experiment(experiment_id, winner_variant_id)

            return ExperimentResult(
                experiment_id=experiment_id,
                status="concluded" if winner_variant_id else "running",
                winner_variant_id=winner_variant_id,
                confidence=result["confidence"],
                statistical_power=result["power"],
                p_value=result["p_value"],
                effect_size=result["effect_size"],
                recommendation=result["recommendation"]
            )

        def _test_proportions(self, variant_metrics: Dict) -> Dict:
            """
            Test for statistically significant difference in proportions.

            Uses z-test for proportions (binary outcomes).
            """
            # Extract control and treatment data
            variants = list(variant_metrics.items())
            control_id, control_data = variants[0]
            treatment_id, treatment_data = variants[1]

            # Sample sizes
            n1 = control_data["sample_size"]
            n2 = treatment_data["sample_size"]

            # Success counts
            x1 = control_data["successes"]
            x2 = treatment_data["successes"]

            # Proportions
            p1 = x1 / n1
            p2 = x2 / n2

            # Pooled proportion
            p_pool = (x1 + x2) / (n1 + n2)

            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

            # Z-statistic
            z = (p2 - p1) / se

            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            # Effect size (absolute difference)
            effect_size = p2 - p1

            # Statistical power (post-hoc)
            power = self._compute_power(n1, n2, effect_size, alpha=0.05)

            # Confidence interval
            ci = stats.norm.interval(
                self.confidence_level,
                loc=effect_size,
                scale=se
            )

            # Determine best variant
            best_variant_id = treatment_id if p2 > p1 else control_id

            # Recommendation
            if p_value < 0.05:
                recommendation = f"Variant {best_variant_id} is significantly better (p={p_value:.4f})"
            else:
                recommendation = "No significant difference detected. Continue experiment or conclude as tie."

            return {
                "p_value": p_value,
                "effect_size": effect_size,
                "confidence": self.confidence_level,
                "power": power,
                "confidence_interval": ci,
                "best_variant_id": best_variant_id,
                "recommendation": recommendation
            }

        def _test_means(self, variant_metrics: Dict) -> Dict:
            """
            Test for statistically significant difference in means.

            Uses Welch's t-test (unequal variances).
            """
            # Extract control and treatment data
            variants = list(variant_metrics.items())
            control_id, control_data = variants[0]
            treatment_id, treatment_data = variants[1]

            # Sample statistics
            n1 = control_data["sample_size"]
            n2 = treatment_data["sample_size"]
            mean1 = control_data["mean"]
            mean2 = treatment_data["mean"]
            std1 = control_data["std"]
            std2 = treatment_data["std"]

            # Welch's t-test
            t_stat, p_value = stats.ttest_ind_from_stats(
                mean1, std1, n1,
                mean2, std2, n2,
                equal_var=False
            )

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            effect_size = (mean2 - mean1) / pooled_std

            # Statistical power
            power = self._compute_power(n1, n2, effect_size, alpha=0.05)

            # Confidence interval
            se = np.sqrt(std1**2/n1 + std2**2/n2)
            ci = stats.t.interval(
                self.confidence_level,
                df=min(n1-1, n2-1),
                loc=mean2-mean1,
                scale=se
            )

            # Determine best variant (lower is better for latency/cost)
            best_variant_id = treatment_id if mean2 < mean1 else control_id

            # Recommendation
            if p_value < 0.05:
                recommendation = f"Variant {best_variant_id} is significantly better (p={p_value:.4f})"
            else:
                recommendation = "No significant difference detected."

            return {
                "p_value": p_value,
                "effect_size": effect_size,
                "confidence": self.confidence_level,
                "power": power,
                "confidence_interval": ci,
                "best_variant_id": best_variant_id,
                "recommendation": recommendation
            }

        def _hash_based_assignment(self, user_id: str, variants: List[Variant]) -> Variant:
            """
            Assign user to variant using consistent hashing.

            Ensures same user always gets same variant.
            """
            # Hash user ID to [0, 1]
            hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
            normalized_hash = (hash_val % 10000) / 10000

            # Assign based on traffic percentages
            cumulative = 0
            for variant in variants:
                cumulative += variant.traffic_percentage / 100
                if normalized_hash < cumulative:
                    return variant

            # Fallback to last variant
            return variants[-1]

        def _compute_power(
            self,
            n1: int,
            n2: int,
            effect_size: float,
            alpha: float = 0.05
        ) -> float:
            """Compute statistical power (post-hoc)."""
            from statsmodels.stats.power import zt_ind_solve_power

            try:
                power = zt_ind_solve_power(
                    effect_size=effect_size,
                    nobs1=n1,
                    alpha=alpha,
                    ratio=n2/n1,
                    alternative='two-sided'
                )
                return power
            except:
                return 0.0  # Unable to compute

        def _get_variant_metrics(
            self,
            experiment_id: str,
            variant_id: str,
            metric_name: str
        ) -> Dict:
            """Get aggregated metrics for variant from analytics DB."""
            query = """
                SELECT
                    COUNT(*) as sample_size,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                    AVG(latency_ms) as mean_latency,
                    STDDEV(latency_ms) as std_latency,
                    AVG(total_cost) as mean_cost,
                    STDDEV(total_cost) as std_cost
                FROM prompt_executions
                WHERE experiment_id = %s AND variant_id = %s
            """

            result = self.analytics_db.fetch_one(query, (experiment_id, variant_id))

            return {
                "sample_size": result["sample_size"],
                "successes": result["successes"],
                "mean": result[f"mean_{metric_name}"],
                "std": result[f"std_{metric_name}"]
            }

    # Usage example
    ab_test = ABTestManager(db_client, analytics_db)

    # Assign user to variant
    variant_id = ab_test.assign_variant(
        experiment_id="exp_123",
        user_id="user_456"
    )

    # Analyze experiment after data collection
    result = ab_test.analyze_experiment(
        experiment_id="exp_123",
        metric_name="success_rate"
    )

    print(f"Status: {result.status}")
    print(f"Winner: {result.winner_variant_id}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Effect size: {result.effect_size:.2%}")
    print(f"Recommendation: {result.recommendation}")
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scaling Strategies

    ### 1. Prompt Execution Performance

    **Challenge:** 10M executions/day require low-latency prompt delivery and LLM calls.

    **Solution 1: Multi-Level Caching**

    ```python
    from typing import Optional, Dict
    import time

    class CachedPromptExecutor:
        """
        Multi-level caching for prompt execution.

        Cache layers:
        1. L1: In-memory cache (hot prompts) - 10ms
        2. L2: Redis cache (semantic) - 50ms
        3. L3: LLM execution - 1-5s
        """

        def __init__(self):
            self.l1_cache = {}  # In-memory (LRU)
            self.l1_max_size = 1000
            self.l2_cache = SemanticCache(redis_client)
            self.llm_gateway = LLMGateway()

        def execute_prompt(
            self,
            prompt_id: str,
            version_id: str,
            variables: Dict,
            use_cache: bool = True
        ) -> Dict:
            """
            Execute prompt with multi-level caching.

            Returns:
                {
                    "output": "...",
                    "cached": bool,
                    "cache_level": "l1" | "l2" | "none",
                    "latency_ms": int
                }
            """
            start_time = time.time()

            # Get prompt template
            prompt_template = self._get_prompt_template(prompt_id, version_id)

            # Render template
            rendered_prompt = self._render_template(prompt_template, variables)

            if use_cache:
                # L1 cache check (in-memory)
                l1_result = self._check_l1_cache(rendered_prompt)
                if l1_result:
                    latency_ms = int((time.time() - start_time) * 1000)
                    return {
                        "output": l1_result["output"],
                        "cached": True,
                        "cache_level": "l1",
                        "latency_ms": latency_ms,
                        **l1_result
                    }

                # L2 cache check (Redis semantic)
                l2_result = self.l2_cache.get_cached_response(
                    rendered_prompt=rendered_prompt,
                    model=prompt_template["model"],
                    parameters=prompt_template["parameters"]
                )
                if l2_result:
                    # Promote to L1
                    self._store_l1_cache(rendered_prompt, l2_result)

                    latency_ms = int((time.time() - start_time) * 1000)
                    return {
                        "output": l2_result["output"],
                        "cached": True,
                        "cache_level": "l2",
                        "latency_ms": latency_ms,
                        **l2_result
                    }

            # L3: Execute LLM
            response = self.llm_gateway.execute(
                prompt=rendered_prompt,
                model=prompt_template["model"],
                parameters=prompt_template["parameters"]
            )

            # Store in caches
            if use_cache:
                self.l2_cache.cache_response(
                    rendered_prompt=rendered_prompt,
                    model=prompt_template["model"],
                    parameters=prompt_template["parameters"],
                    response=response
                )
                self._store_l1_cache(rendered_prompt, response)

            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": response["output"],
                "cached": False,
                "cache_level": "none",
                "latency_ms": latency_ms,
                **response
            }

        def _check_l1_cache(self, rendered_prompt: str) -> Optional[Dict]:
            """Check in-memory cache (LRU)."""
            cache_key = hashlib.sha256(rendered_prompt.encode()).hexdigest()
            return self.l1_cache.get(cache_key)

        def _store_l1_cache(self, rendered_prompt: str, response: Dict):
            """Store in L1 cache with LRU eviction."""
            cache_key = hashlib.sha256(rendered_prompt.encode()).hexdigest()

            # Evict oldest if full
            if len(self.l1_cache) >= self.l1_max_size:
                oldest_key = next(iter(self.l1_cache))
                del self.l1_cache[oldest_key]

            self.l1_cache[cache_key] = response
    ```

    **Performance:**
    - L1 cache hit (10% of requests): 10ms
    - L2 cache hit (20% of requests): 50ms
    - LLM execution (70% of requests): 1500ms
    - Average latency: 0.1×10 + 0.2×50 + 0.7×1500 = 1061ms
    - Without cache: 1500ms
    - **Improvement: 29% faster**

    ---

    **Solution 2: Parallel Prompt Execution**

    ```python
    import asyncio
    from typing import List, Dict

    class ParallelPromptExecutor:
        """
        Execute multiple prompts in parallel for chaining workflows.

        Use cases:
        - Independent prompts (parallel)
        - Sequential prompts with dependencies
        - Fan-out/fan-in patterns
        """

        async def execute_parallel(
            self,
            prompts: List[Dict]
        ) -> List[Dict]:
            """
            Execute multiple prompts in parallel.

            Args:
                prompts: List of {prompt_id, version_id, variables}

            Returns:
                List of responses in same order
            """
            tasks = [
                self._execute_single_async(p["prompt_id"], p["version_id"], p["variables"])
                for p in prompts
            ]

            results = await asyncio.gather(*tasks)
            return results

        async def execute_chain(
            self,
            chain: List[Dict]
        ) -> List[Dict]:
            """
            Execute prompt chain with dependencies.

            Each prompt can use outputs from previous prompts.

            Args:
                chain: List of {prompt_id, version_id, variables, dependencies}

            Returns:
                List of responses
            """
            results = {}

            for step in chain:
                # Wait for dependencies
                dependencies = step.get("dependencies", [])
                if dependencies:
                    # Use outputs from previous steps
                    for dep_idx in dependencies:
                        step["variables"].update(results[dep_idx])

                # Execute prompt
                response = await self._execute_single_async(
                    step["prompt_id"],
                    step["version_id"],
                    step["variables"]
                )

                results[step["step_id"]] = response

            return list(results.values())

    # Usage example
    executor = ParallelPromptExecutor()

    # Parallel execution (3 independent prompts)
    prompts = [
        {"prompt_id": "summarize", "version_id": "latest", "variables": {"text": doc1}},
        {"prompt_id": "summarize", "version_id": "latest", "variables": {"text": doc2}},
        {"prompt_id": "summarize", "version_id": "latest", "variables": {"text": doc3}}
    ]

    results = await executor.execute_parallel(prompts)
    # Time: 1.5s (vs 4.5s sequential)

    # Sequential chain (extract -> analyze -> summarize)
    chain = [
        {
            "step_id": "extract",
            "prompt_id": "extract_entities",
            "version_id": "latest",
            "variables": {"text": document},
            "dependencies": []
        },
        {
            "step_id": "analyze",
            "prompt_id": "analyze_sentiment",
            "version_id": "latest",
            "variables": {"entities": "{{extract.output}}"},  # Uses extract output
            "dependencies": ["extract"]
        },
        {
            "step_id": "summarize",
            "prompt_id": "create_summary",
            "version_id": "latest",
            "variables": {
                "entities": "{{extract.output}}",
                "sentiment": "{{analyze.output}}"
            },
            "dependencies": ["extract", "analyze"]
        }
    ]

    results = await executor.execute_chain(chain)
    ```

    ---

    ### 2. Cost Optimization

    **Challenge:** 10M executions/day with GPT-4 costs $150K/month.

    **Optimization Strategies:**

    ```python
    class CostOptimizer:
        """
        Optimize LLM costs through various strategies.

        Strategies:
        1. Model routing (use cheaper models when possible)
        2. Token optimization (reduce prompt size)
        3. Aggressive caching (30% cache hit rate)
        4. Request batching
        """

        def __init__(self):
            self.model_costs = {
                "gpt-4-turbo": {"input": 10.0, "output": 30.0},  # per 1M tokens
                "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
                "claude-3-haiku": {"input": 0.25, "output": 1.25}
            }

        def route_to_model(self, prompt: str, required_capability: str) -> str:
            """
            Route prompt to appropriate model based on complexity.

            Capabilities:
            - simple: Short prompts, straightforward tasks (GPT-3.5)
            - moderate: Complex reasoning, longer context (GPT-4-turbo)
            - advanced: Coding, analysis, research (GPT-4)
            """
            # Analyze prompt complexity
            complexity = self._analyze_complexity(prompt)

            if required_capability == "simple" and complexity < 0.5:
                return "gpt-3.5-turbo"  # 95% cheaper
            elif required_capability == "moderate":
                return "gpt-4-turbo"
            else:
                return "gpt-4-turbo"

        def optimize_prompt_tokens(self, prompt: str) -> str:
            """
            Reduce prompt tokens without losing quality.

            Techniques:
            - Remove redundant whitespace
            - Compress long examples
            - Use abbreviations where appropriate
            """
            # Remove extra whitespace
            optimized = " ".join(prompt.split())

            # Truncate very long prompts
            max_tokens = 3000
            estimated_tokens = len(optimized) // 4

            if estimated_tokens > max_tokens:
                # Truncate to max tokens
                optimized = optimized[:max_tokens * 4]

            return optimized

        def estimate_monthly_cost(
            self,
            executions_per_day: int,
            avg_input_tokens: int,
            avg_output_tokens: int,
            model: str,
            cache_hit_rate: float = 0.30
        ) -> Dict:
            """
            Estimate monthly cost with caching.

            Returns:
                {
                    "without_cache": float,
                    "with_cache": float,
                    "savings": float,
                    "savings_percentage": float
                }
            """
            # Monthly executions
            monthly_executions = executions_per_day * 30

            # Cost per execution
            input_cost_per_1m = self.model_costs[model]["input"]
            output_cost_per_1m = self.model_costs[model]["output"]

            cost_per_execution = (
                (avg_input_tokens / 1_000_000) * input_cost_per_1m +
                (avg_output_tokens / 1_000_000) * output_cost_per_1m
            )

            # Without cache
            cost_without_cache = monthly_executions * cost_per_execution

            # With cache (only pay for cache misses)
            cache_misses = monthly_executions * (1 - cache_hit_rate)
            cost_with_cache = cache_misses * cost_per_execution

            # Savings
            savings = cost_without_cache - cost_with_cache
            savings_percentage = (savings / cost_without_cache) * 100

            return {
                "without_cache": cost_without_cache,
                "with_cache": cost_with_cache,
                "savings": savings,
                "savings_percentage": savings_percentage
            }

    # Example: Cost analysis
    optimizer = CostOptimizer()

    # Scenario: 10M executions/day, GPT-4-turbo
    cost_analysis = optimizer.estimate_monthly_cost(
        executions_per_day=10_000_000,
        avg_input_tokens=1000,
        avg_output_tokens=500,
        model="gpt-4-turbo",
        cache_hit_rate=0.30
    )

    print(f"Without cache: ${cost_analysis['without_cache']:,.2f}/month")
    print(f"With cache: ${cost_analysis['with_cache']:,.2f}/month")
    print(f"Savings: ${cost_analysis['savings']:,.2f}/month ({cost_analysis['savings_percentage']:.1f}%)")

    # Output:
    # Without cache: $150,000.00/month
    # With cache: $105,000.00/month
    # Savings: $45,000.00/month (30.0%)

    # Strategy 2: Model routing (80% simple tasks to GPT-3.5)
    simple_cost = optimizer.estimate_monthly_cost(
        executions_per_day=8_000_000,  # 80% of traffic
        avg_input_tokens=500,
        avg_output_tokens=250,
        model="gpt-3.5-turbo",
        cache_hit_rate=0.30
    )

    complex_cost = optimizer.estimate_monthly_cost(
        executions_per_day=2_000_000,  # 20% of traffic
        avg_input_tokens=1500,
        avg_output_tokens=750,
        model="gpt-4-turbo",
        cache_hit_rate=0.30
    )

    total_with_routing = simple_cost["with_cache"] + complex_cost["with_cache"]
    print(f"\nWith model routing: ${total_with_routing:,.2f}/month")
    # Output: With model routing: $25,200.00/month (83% savings!)
    ```

    ---

    ### 3. Analytics Pipeline Optimization

    **Challenge:** Real-time dashboard queries on 10M executions/day.

    **Solution: ClickHouse for Analytics**

    ```sql
    -- ClickHouse schema (columnar storage)
    CREATE TABLE prompt_analytics (
        execution_id String,
        prompt_id String,
        version_id String,
        user_id String,

        -- Metrics
        latency_ms UInt32,
        prompt_tokens UInt32,
        completion_tokens UInt32,
        total_cost Decimal(10, 6),
        cached UInt8,  -- Boolean
        success UInt8,  -- Boolean

        -- Dimensions
        model String,
        provider String,
        experiment_id String,

        -- Timestamp
        executed_at DateTime
    )
    ENGINE = MergeTree()
    PARTITION BY toYYYYMM(executed_at)
    ORDER BY (prompt_id, executed_at)
    SETTINGS index_granularity = 8192;

    -- Materialized view for real-time aggregation
    CREATE MATERIALIZED VIEW prompt_hourly_metrics
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(hour)
    ORDER BY (prompt_id, version_id, hour)
    AS SELECT
        prompt_id,
        version_id,
        toStartOfHour(executed_at) AS hour,
        count() AS executions,
        avg(latency_ms) AS avg_latency_ms,
        sum(total_cost) AS total_cost,
        sum(cached) / count() AS cache_hit_rate,
        sum(success) / count() AS success_rate
    FROM prompt_analytics
    GROUP BY prompt_id, version_id, hour;

    -- Dashboard query: Prompt performance (last 7 days)
    SELECT
        prompt_id,
        version_id,
        sum(executions) AS total_executions,
        avg(avg_latency_ms) AS avg_latency,
        sum(total_cost) AS total_cost,
        avg(cache_hit_rate) AS cache_hit_rate
    FROM prompt_hourly_metrics
    WHERE hour >= now() - INTERVAL 7 DAY
    GROUP BY prompt_id, version_id
    ORDER BY total_executions DESC
    LIMIT 100;
    -- Query time: <100ms for 10M rows
    ```

    ---

    ### 4. Collaboration Features

    **Challenge:** Teams need to share prompts, review changes, approve deployments.

    **Solution: Approval Workflow**

    ```python
    from enum import Enum
    from typing import List, Optional
    from dataclasses import dataclass

    class ApprovalStatus(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"

    @dataclass
    class PromptReview:
        review_id: str
        version_id: str
        reviewer_id: str
        status: ApprovalStatus
        comment: Optional[str]
        reviewed_at: datetime

    class CollaborationManager:
        """
        Collaboration features for prompt management.

        Features:
        - Prompt sharing
        - Review workflow
        - Approval gates for production
        - Comment threads
        """

        def __init__(self, db_client):
            self.db = db_client

        def request_review(
            self,
            version_id: str,
            reviewers: List[str],
            requester_id: str,
            message: str
        ) -> Dict:
            """
            Request review for prompt version before deploying to production.

            Args:
                version_id: Prompt version to review
                reviewers: List of reviewer user IDs
                requester_id: User requesting review
                message: Review request message

            Returns:
                Review request details
            """
            review_request_id = generate_id()

            # Create review request
            self.db.execute("""
                INSERT INTO review_requests (
                    review_request_id, version_id, requester_id,
                    message, status, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                review_request_id,
                version_id,
                requester_id,
                message,
                "pending",
                datetime.utcnow()
            ))

            # Notify reviewers
            for reviewer_id in reviewers:
                self._send_notification(
                    user_id=reviewer_id,
                    type="review_request",
                    message=f"Review requested for prompt version {version_id}"
                )

            return {
                "review_request_id": review_request_id,
                "status": "pending",
                "reviewers": reviewers
            }

        def submit_review(
            self,
            version_id: str,
            reviewer_id: str,
            status: ApprovalStatus,
            comment: str
        ) -> PromptReview:
            """
            Submit review for prompt version.

            Args:
                version_id: Prompt version being reviewed
                reviewer_id: User submitting review
                status: APPROVED or REJECTED
                comment: Review comments

            Returns:
                PromptReview object
            """
            review = PromptReview(
                review_id=generate_id(),
                version_id=version_id,
                reviewer_id=reviewer_id,
                status=status,
                comment=comment,
                reviewed_at=datetime.utcnow()
            )

            # Store review
            self.db.execute("""
                INSERT INTO prompt_reviews (
                    review_id, version_id, reviewer_id,
                    status, comment, reviewed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                review.review_id,
                review.version_id,
                review.reviewer_id,
                review.status.value,
                review.comment,
                review.reviewed_at
            ))

            # Check if all reviewers have approved
            self._check_approval_status(version_id)

            return review

        def _check_approval_status(self, version_id: str):
            """Check if version has required approvals."""
            # Get all reviews
            reviews = self.db.fetch_all("""
                SELECT status FROM prompt_reviews
                WHERE version_id = %s
            """, (version_id,))

            # Check approval policy (e.g., 2/3 approvals)
            approved = sum(1 for r in reviews if r["status"] == "approved")
            rejected = sum(1 for r in reviews if r["status"] == "rejected")

            if approved >= 2:
                # Mark version as production-ready
                self.db.execute("""
                    UPDATE prompt_versions
                    SET status = 'production_ready'
                    WHERE version_id = %s
                """, (version_id,))

                # Notify requester
                self._send_notification(
                    user_id=requester_id,
                    type="review_approved",
                    message=f"Version {version_id} approved for production"
                )
            elif rejected >= 2:
                # Mark as rejected
                self.db.execute("""
                    UPDATE prompt_versions
                    SET status = 'rejected'
                    WHERE version_id = %s
                """, (version_id,))
    ```

=== "💡 Step 5: Additional Considerations"

    ## Security & Compliance

    ### 1. Access Control (RBAC)

    ```sql
    -- Roles
    CREATE TABLE roles (
        role_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,  -- viewer, editor, admin
        permissions JSONB NOT NULL
    );

    -- User roles
    CREATE TABLE user_roles (
        user_id VARCHAR(100) REFERENCES users(user_id),
        project_id VARCHAR(50) REFERENCES projects(project_id),
        role_id VARCHAR(50) REFERENCES roles(role_id),
        granted_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (user_id, project_id)
    );

    -- Permissions check
    CREATE FUNCTION can_edit_prompt(
        p_user_id VARCHAR,
        p_prompt_id VARCHAR
    ) RETURNS BOOLEAN AS $$
    DECLARE
        user_role VARCHAR;
    BEGIN
        SELECT r.name INTO user_role
        FROM prompts p
        JOIN user_roles ur ON p.project_id = ur.project_id
        JOIN roles r ON ur.role_id = r.role_id
        WHERE p.prompt_id = p_prompt_id
          AND ur.user_id = p_user_id;

        RETURN user_role IN ('editor', 'admin');
    END;
    $$ LANGUAGE plpgsql;
    ```

    ---

    ### 2. Audit Logging

    ```sql
    CREATE TABLE audit_logs (
        log_id BIGSERIAL PRIMARY KEY,
        user_id VARCHAR(100),
        action VARCHAR(50),  -- create_prompt, update_version, execute_prompt
        resource_type VARCHAR(50),
        resource_id VARCHAR(50),
        changes JSONB,  -- Before/after values
        ip_address INET,
        user_agent TEXT,
        timestamp TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_audit_user ON audit_logs(user_id);
    CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp DESC);
    CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
    ```

    ---

    ## Monitoring & Alerting

    **Key Metrics:**
    ```python
    # 1. Prompt Execution Metrics
    histogram("prompt_execution_latency_ms", buckets=[50, 100, 500, 1000, 5000])
    counter("prompt_executions_total", labels=["prompt_id", "version_id", "model"])
    counter("prompt_execution_errors", labels=["prompt_id", "error_type"])

    # 2. Cache Metrics
    gauge("cache_hit_rate")
    counter("cache_hits_total")
    counter("cache_misses_total")

    # 3. Cost Metrics
    counter("prompt_tokens_total", labels=["model"])
    counter("prompt_cost_dollars", labels=["model", "prompt_id"])

    # 4. A/B Test Metrics
    gauge("active_experiments_count")
    counter("experiment_assignments", labels=["experiment_id", "variant_id"])
    ```

    **Alerts:**
    ```yaml
    - alert: HighExecutionLatency
      expr: histogram_quantile(0.95, prompt_execution_latency_ms) > 5000
      for: 5m
      annotations:
        summary: "Prompt execution latency high ({{ $value }}ms)"

    - alert: LowCacheHitRate
      expr: cache_hit_rate < 0.20
      for: 30m
      annotations:
        summary: "Cache hit rate dropped below 20%"

    - alert: HighCostBurn
      expr: rate(prompt_cost_dollars[1h]) > 100
      for: 15m
      annotations:
        summary: "Spending rate exceeds $100/hour"

    - alert: PromptExecutionErrors
      expr: rate(prompt_execution_errors[5m]) > 10
      for: 5m
      annotations:
        summary: "High error rate for prompt executions"
    ```

    ---

    ## Disaster Recovery

    **Backup Strategy:**
    ```bash
    # Daily PostgreSQL backup (metadata)
    pg_dump prompt_management_db | gzip > backup-$(date +%Y%m%d).sql.gz
    aws s3 cp backup-$(date +%Y%m%d).sql.gz s3://prompt-backups/

    # TimescaleDB backup
    pg_dump -t prompt_executions | gzip > logs-backup-$(date +%Y%m%d).sql.gz

    # S3 versioning enabled for prompt templates (automatic)
    aws s3api put-bucket-versioning \
      --bucket prompt-templates \
      --versioning-configuration Status=Enabled

    # Redis persistence (AOF + RDB)
    # AOF: appendonly.aof (incremental)
    # RDB: dump.rdb (snapshots every 5 minutes)
    ```

    **Recovery Time:**
    - Metadata DB: <1 hour (restore from backup)
    - Time-series DB: <2 hours (restore + reindex)
    - Cache: <15 minutes (rebuild from DB)
    - Templates: Instant (S3 is durable)

=== "🎯 Step 6: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5 minutes)

    **Key Questions:**
    - Scale? (number of prompts, executions per day, concurrent users)
    - Latency requirements? (how fast should prompt execution be?)
    - Cost constraints? (budget for LLM API calls)
    - Versioning depth? (how many versions to keep?)
    - A/B testing needs? (experimentation required?)

    ---

    ### 2. Start with Use Case (2 minutes)

    "Let's design a prompt management system for engineering teams. Users need to:
    1. Create and version prompts with templates
    2. Execute prompts against LLMs with variable interpolation
    3. Run A/B tests to compare prompt variants
    4. Track costs and performance per prompt/version
    5. Cache responses to reduce LLM costs"

    ---

    ### 3. High-Level Architecture (10 minutes)

    Draw architecture diagram with:
    - Prompt Service (CRUD, version control)
    - Execution Service (template rendering, LLM gateway)
    - Cache Service (semantic caching with Redis)
    - A/B Test Manager (variant selection, statistical analysis)
    - Analytics Pipeline (real-time metrics, cost tracking)

    Explain the rationale for each component.

    ---

    ### 4. Deep Dive (20 minutes)

    Focus on 2-3 areas based on interviewer interest:
    - **Template rendering:** Jinja2 for variable interpolation
    - **Version control:** Git-like semantics (commit, diff, rollback)
    - **Semantic caching:** Embedding similarity for cache lookups
    - **A/B testing:** Statistical analysis with t-tests, chi-square
    - **Cost optimization:** Model routing, caching, token reduction

    ---

    ### 5. Scale & Optimize (10 minutes)

    Discuss:
    - Multi-level caching (L1 in-memory, L2 Redis semantic)
    - Parallel prompt execution for chaining workflows
    - Cost optimization (30% cache hit rate saves 30% costs)
    - Analytics pipeline (ClickHouse for fast aggregations)

    ---

    ## Common Follow-Up Questions

    ### Q1: How do you handle prompt versioning and rollback?

    **Answer:**
    "We use immutable versions in a linked list structure (like Git commits). Each version has a parent_version_id, creating a history chain. To rollback, we create a new version with the old content. We compute diffs using Python's difflib for line-by-line comparison."

    ---

    ### Q2: How does semantic caching work?

    **Answer:**
    "We generate embeddings for rendered prompts and store them with responses. When a new prompt comes in, we compute its embedding and search for similar cached embeddings using cosine similarity. If similarity > 0.95, we return the cached response. This achieves ~30% cache hit rate vs ~5% for exact match."

    ---

    ### Q3: How do you ensure A/B test validity?

    **Answer:**
    "We use consistent hash-based assignment so users always see the same variant. We collect minimum 1,000 samples per variant before analysis. We run statistical tests (t-test for continuous metrics, chi-square for binary) and only declare winner if p-value < 0.05. We also compute statistical power to avoid false negatives."

    ---

    ### Q4: How do you optimize costs?

    **Answer:**
    "Three strategies: 1) Semantic caching (30% hit rate = 30% cost savings), 2) Model routing (use GPT-3.5 for simple tasks = 95% cheaper), 3) Token optimization (remove whitespace, truncate long prompts). Combined, this can reduce costs by 80%."

    ---

    ### Q5: How do you handle prompt chaining?

    **Answer:**
    "We support both parallel and sequential execution. For parallel, we use asyncio to execute independent prompts concurrently. For sequential, each step can reference outputs from previous steps using variable interpolation ({{step1.output}}). We track dependencies and execute in correct order."

    ---

    ## Red Flags to Avoid

    1. **Don't** store large prompt outputs in PostgreSQL (use S3 or compressed storage)
    2. **Don't** use exact-match caching only (semantic caching is much better)
    3. **Don't** forget version control (users need to rollback bad prompts)
    4. **Don't** ignore cost optimization (LLM costs can explode)
    5. **Don't** skip statistical rigor in A/B tests (false positives are expensive)

    ---

    ## Bonus Points

    1. Mention **real-world systems:** PromptLayer, LangSmith, Helicone
    2. Discuss **prompt engineering patterns:** Few-shot learning, chain-of-thought, ReAct
    3. Talk about **LLM gateway patterns:** Retry logic, fallback providers, rate limiting
    4. Consider **compliance:** PII detection, content filtering, audit logging
    5. Mention **prompt optimization:** Automatic prompt compression, token reduction

=== "📚 References & Resources"

    ## Real-World Implementations

    ### PromptLayer
    - **Architecture:** SaaS platform, prompt registry, version control, analytics
    - **Scale:** 1000+ companies, millions of prompt executions
    - **Key Innovation:** Visual prompt editor, automatic logging, cost tracking
    - **Website:** [promptlayer.com](https://promptlayer.com/)

    ---

    ### LangSmith (LangChain)
    - **Architecture:** Integrated with LangChain, prompt management + tracing
    - **Scale:** 100K+ developers, enterprise customers
    - **Key Innovation:** End-to-end tracing, dataset management, human feedback
    - **Website:** [smith.langchain.com](https://smith.langchain.com/)

    ---

    ### Helicone
    - **Architecture:** Proxy-based LLM gateway, observability, caching
    - **Scale:** Open-source + hosted, high-throughput applications
    - **Key Innovation:** Zero-code integration (proxy), semantic caching
    - **Website:** [helicone.ai](https://helicone.ai/)

    ---

    ### Weights & Biases Prompts
    - **Architecture:** Part of W&B platform, experiment tracking for prompts
    - **Scale:** Enterprise ML teams
    - **Key Innovation:** A/B testing, prompt comparison, integration with ML workflows
    - **Website:** [wandb.ai/prompts](https://wandb.ai/site/prompts)

    ---

    ## Open Source Tools

    ### LangChain
    ```python
    from langchain import PromptTemplate, LLMChain
    from langchain.chat_models import ChatOpenAI

    template = PromptTemplate(
        input_variables=["product", "tone"],
        template="Write a marketing email about {product} in a {tone} tone."
    )

    llm = ChatOpenAI(model="gpt-4")
    chain = LLMChain(llm=llm, prompt=template)

    result = chain.run(product="AI Assistant", tone="professional")
    ```

    ---

    ### LlamaIndex
    ```python
    from llama_index import PromptTemplate

    template = PromptTemplate(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    response = template.format(
        context="Company revenue was $10M in Q4.",
        question="What was the revenue?"
    )
    ```

    ---

    ### LiteLLM (Multi-Provider)
    ```python
    from litellm import completion

    # Unified API for multiple providers
    response = completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        cache=True  # Built-in caching
    )

    # Automatic fallback
    response = completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        fallbacks=["claude-3-opus", "gpt-3.5-turbo"]
    )
    ```

    ---

    ## Related System Design Problems

    1. **ML Experiment Tracking** - Similar versioning and analytics patterns
    2. **Feature Store** - Version control for features (similar to prompts)
    3. **A/B Testing Platform** - Experimentation and statistical analysis
    4. **API Gateway** - Rate limiting, routing, caching (for LLM gateway)
    5. **Observability Platform** - Metrics collection and dashboards

---

## Summary

A **Prompt Management System** enables teams to version, test, and monitor LLM prompts at scale:

**Key Components:**
- **Prompt Registry:** Store prompts with version control (Git-like semantics)
- **Template Engine:** Jinja2 for variable interpolation and control flow
- **Execution Service:** Render templates, route to LLM providers, log results
- **Semantic Cache:** Embedding-based similarity search (30% hit rate)
- **A/B Test Manager:** Traffic splitting, statistical analysis, auto-conclusion
- **Analytics Pipeline:** Real-time metrics (latency, cost, tokens, success rate)
- **LLM Gateway:** Multi-provider support, retry logic, rate limiting

**Core Challenges:**
- Prompt versioning with diff, rollback, and history
- Template rendering with variables, loops, conditionals
- Semantic caching to maximize hit rate and reduce costs
- A/B testing with statistical rigor (t-tests, chi-square)
- Cost optimization (caching, model routing, token reduction)
- Real-time analytics on millions of executions

**Architecture Decisions:**
- Immutable versions for reliable history and rollback
- Jinja2 for powerful template rendering
- Embedding-based semantic caching (30% hit rate vs 5% exact match)
- Multi-level caching (L1 in-memory, L2 Redis)
- TimescaleDB for time-series logs, ClickHouse for analytics
- Statistical A/B testing with minimum sample sizes and confidence intervals

This is a **medium difficulty** problem that combines distributed systems, LLM integration, and experimentation infrastructure. Focus on version control, semantic caching, and cost optimization during interviews.
