# Design ChatGPT-like System (LLM Chat Platform)

A large language model (LLM) powered conversational AI platform that enables users to have multi-turn conversations with AI assistants, supporting streaming responses, context management, rate limiting, and plugin/tool integration.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M daily active users, 2B requests/day, 500B tokens/day, 1M requests/min peak |
| **Key Challenges** | LLM serving at scale, streaming responses, context window management, GPU optimization, rate limiting |
| **Core Concepts** | Transformer models, vLLM, batching, KV cache, SSE/WebSocket, prompt caching, token bucket |
| **Companies** | OpenAI (ChatGPT), Anthropic (Claude), Google (Bard/Gemini), Microsoft (Copilot), Meta (Llama) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Chat Completions** | Send message, receive AI response with streaming | P0 (Must have) |
    | **Multi-turn Conversations** | Maintain context across multiple messages | P0 (Must have) |
    | **Streaming Responses** | Stream tokens as generated (SSE/WebSocket) | P0 (Must have) |
    | **Conversation History** | Save, retrieve, continue past conversations | P0 (Must have) |
    | **Rate Limiting** | Token-based quotas per user/organization | P0 (Must have) |
    | **Model Selection** | Choose between different models (GPT-4, GPT-3.5, etc.) | P0 (Must have) |
    | **System Prompts** | Custom instructions for AI behavior | P1 (Should have) |
    | **Tool/Function Calling** | Enable AI to call external APIs/tools | P1 (Should have) |
    | **Content Moderation** | Filter harmful/unsafe content input/output | P1 (Should have) |
    | **Usage Tracking** | Track tokens consumed for billing | P1 (Should have) |
    | **Multi-modal Support** | Handle images, files alongside text | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (assume pre-trained models)
    - Fine-tuning pipelines
    - Data annotation/labeling
    - Model evaluation frameworks
    - Advanced voice/video capabilities
    - Web browsing/real-time data retrieval

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (TTFT)** | < 500ms p95 time-to-first-token | Users expect immediate response start |
    | **Latency (Token Throughput)** | > 30 tokens/sec per request | Smooth streaming experience |
    | **Availability** | 99.9% uptime (SLA) | Critical for production applications |
    | **Consistency** | Strong consistency for conversations | User must see all their messages |
    | **Scalability** | Handle 10x traffic spikes | Viral features, new model launches |
    | **GPU Utilization** | > 70% average utilization | GPUs expensive, maximize efficiency |
    | **Cost Optimization** | < $0.10 per 1M tokens (infra cost) | Keep margins healthy with efficient serving |
    | **Security** | Multi-tenant isolation, no data leakage | Protect user privacy and conversations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 100M
    Monthly Active Users (MAU): 200M

    Chat requests:
    - Average conversations per DAU: 5 conversations/day
    - Messages per conversation: 8 messages (4 turns)
    - Daily requests: 100M × 5 × 4 = 2B requests/day
    - Request QPS: 2B / 86,400 = ~23,150 req/sec average
    - Peak QPS: 3x average = ~70,000 req/sec

    Token processing:
    - Average input tokens per request: 1,500 tokens (context + new message)
    - Average output tokens per request: 500 tokens (response)
    - Total tokens per request: 2,000 tokens
    - Daily tokens: 2B requests × 2,000 tokens = 4T tokens/day
    - Token throughput: 4T / 86,400 = ~46M tokens/sec average
    - Peak: ~140M tokens/sec

    Model distribution:
    - GPT-4 class (large): 20% of requests = 4,600 req/sec
    - GPT-3.5 class (medium): 60% of requests = 13,900 req/sec
    - Small models: 20% of requests = 4,600 req/sec

    Streaming:
    - 80% of requests use streaming
    - Average stream duration: 15-20 seconds
    - Concurrent streams: 70,000 × 0.8 × 15 = ~840,000 concurrent streams peak

    Read/Write ratio: 10:1 (conversations read >> write)
    ```

    ### Storage Estimates

    ```
    Conversations:
    - Active conversations: 100M DAU × 5 = 500M conversations/day
    - Conversation metadata: 1 KB (id, user, created_at, title)
    - 500M × 1 KB × 30 days = 15 TB/month

    Messages:
    - Messages per day: 2B messages
    - Average message size: 2 KB (text + metadata)
    - Daily storage: 2B × 2 KB = 4 TB/day
    - Monthly: 4 TB × 30 = 120 TB/month
    - 1 year retention: 120 TB × 12 = 1.44 PB

    Prompt cache:
    - Cached prefixes: 100M active conversations
    - Average cached context: 10 KB per conversation
    - 100M × 10 KB = 1 TB (in-memory/SSD)

    User data:
    - 200M MAU × 5 KB = 1 TB

    Usage/billing records:
    - 2B requests/day × 200 bytes = 400 GB/day
    - 1 year: 400 GB × 365 = 146 TB

    Model weights (storage):
    - Large model (175B params): 350 GB per copy
    - Medium model (7B params): 14 GB per copy
    - 100 replicas × 350 GB = 35 TB (large models)

    Total: 15 TB (conversations) + 1.44 PB (messages) + 1 TB (cache) + 1 TB (users) + 146 TB (billing) + 35 TB (models) ≈ 1.6 PB
    ```

    ### Compute Estimates (GPU)

    ```
    Large model (GPT-4 class, 175B parameters):
    - Requests per second: 4,600 req/sec
    - Tokens per second: 4,600 × 2,000 = 9.2M tokens/sec
    - Throughput per A100 GPU (80GB): ~2,000 tokens/sec (with batching)
    - GPUs needed: 9.2M / 2,000 = 4,600 A100 GPUs
    - With batching efficiency (3x): ~1,500 A100 GPUs

    Medium model (GPT-3.5 class, 7B parameters):
    - Requests per second: 13,900 req/sec
    - Tokens per second: 13,900 × 2,000 = 27.8M tokens/sec
    - Throughput per A100 GPU: ~10,000 tokens/sec
    - GPUs needed: 27.8M / 10,000 = 2,780 A100 GPUs
    - With batching: ~900 A100 GPUs

    Total GPUs: 1,500 + 900 = 2,400 A100 GPUs (80GB)
    Cost: 2,400 × $2.50/hour = $6,000/hour = $144,000/day = $4.3M/month

    GPU Memory requirements:
    - Model weights: 350 GB (large), 14 GB (medium)
    - KV cache per request: 50 MB (average context)
    - Batch size: 32 concurrent requests per GPU
    - Memory per GPU: 350 GB + (32 × 50 MB) = ~352 GB (needs A100 80GB or H100)
    ```

    ### Bandwidth Estimates

    ```
    Request ingress:
    - 23,150 req/sec × 6 KB (average input) = 139 MB/sec ≈ 1.1 Gbps

    Response egress (streaming):
    - 23,150 req/sec × 2 KB (average output) = 46 MB/sec ≈ 368 Mbps
    - Streaming overhead (SSE): +30% = 480 Mbps

    WebSocket connections:
    - Concurrent connections: 840,000 peak
    - Heartbeat overhead: 840K × 100 bytes/10s = 8.4 MB/sec ≈ 67 Mbps

    Internal (model serving):
    - GPU → Cache tier: 46M tokens/sec × 8 bytes = 368 MB/sec ≈ 2.9 Gbps
    - Cache → API tier: 139 MB/sec ≈ 1.1 Gbps

    Total ingress: ~1.1 Gbps
    Total egress: ~550 Mbps
    Internal: ~4 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Prompt cache (KV cache):
    - Active conversations: 10M concurrent
    - KV cache per conversation: 50 MB
    - Total: 10M × 50 MB = 500 TB (distributed across GPUs)

    Response cache (for common queries):
    - Hot responses: 1M cached responses
    - Average size: 5 KB
    - 1M × 5 KB = 5 GB

    User session cache:
    - Active users: 10M concurrent
    - Session data: 10 KB per user
    - 10M × 10 KB = 100 GB

    Rate limit state:
    - Active users/orgs: 50M
    - State per user: 1 KB (token bucket state)
    - 50M × 1 KB = 50 GB

    Conversation metadata cache:
    - Hot conversations: 50M
    - Metadata: 5 KB per conversation
    - 50M × 5 KB = 250 GB

    Total cache: 500 TB (KV) + 5 GB (responses) + 100 GB (sessions) + 50 GB (rate limits) + 250 GB (metadata) ≈ 500 TB
    ```

    ---

    ## Key Assumptions

    1. Average conversation has 8 messages (4 turns)
    2. Average input: 1,500 tokens, output: 500 tokens
    3. 80% of users prefer streaming responses
    4. GPU utilization can reach 70% with good batching
    5. Large models (GPT-4 class) used for 20% of requests
    6. 90% of conversations reuse cached prompts/context
    7. Rate limiting at user and organization level
    8. Multi-tenant system with strict isolation

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separation of concerns:** API gateway, inference, conversation storage independent
    2. **Horizontal scaling:** Scale each component independently (API, GPU pools, storage)
    3. **Streaming-first:** Optimize for low latency token delivery
    4. **Efficient batching:** Maximize GPU utilization via continuous batching
    5. **Prompt caching:** Reuse KV cache for common prefixes to reduce compute
    6. **Multi-tenancy:** Strong isolation between users/organizations

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
            API_Client[API Client<br/>SDKs]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets]
            LB[Load Balancer<br/>L7 with WebSocket support]
        end

        subgraph "API Gateway"
            Auth[Auth Service<br/>JWT validation]
            RateLimit[Rate Limiter<br/>Token bucket per user/org]
            Router[Request Router<br/>Model selection, routing]
        end

        subgraph "Chat API Layer"
            Chat_API[Chat API<br/>REST + WebSocket]
            Stream_Manager[Stream Manager<br/>SSE/WebSocket handler]
            Conv_Service[Conversation Service<br/>CRUD operations]
            Usage_Tracker[Usage Tracker<br/>Token counting, billing]
        end

        subgraph "LLM Inference Layer"
            LLM_Gateway[Inference Gateway<br/>Request batching, scheduling]
            Model_Large[Large Model Pool<br/>GPT-4 class<br/>vLLM servers]
            Model_Medium[Medium Model Pool<br/>GPT-3.5 class<br/>vLLM servers]
            Model_Small[Small Model Pool<br/>Fast models]
            Cache_Manager[Cache Manager<br/>KV cache, prefix caching]
        end

        subgraph "Safety & Tools"
            Moderation[Content Moderation<br/>Input/output filtering]
            Tool_Executor[Tool Executor<br/>Function calling]
            Plugin_Manager[Plugin Manager<br/>External integrations]
        end

        subgraph "Caching"
            Redis_Session[Redis<br/>Session cache]
            Redis_Rate[Redis<br/>Rate limit state]
            Redis_Conv[Redis<br/>Conversation cache]
            Redis_Prompt[Redis<br/>Prompt cache metadata]
        end

        subgraph "Storage"
            Conv_DB[(Conversation DB<br/>PostgreSQL<br/>Sharded by user_id)]
            Message_DB[(Message Store<br/>Cassandra<br/>Time-series)]
            User_DB[(User DB<br/>PostgreSQL<br/>Auth, billing)]
            Usage_DB[(Usage Logs<br/>ClickHouse<br/>Analytics)]
            Model_Store[Model Registry<br/>S3<br/>Model weights]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "Analytics & Monitoring"
            Metrics[Metrics Service<br/>Prometheus/Grafana]
            Logs[Log Aggregation<br/>ELK Stack]
            Alerts[Alerting<br/>PagerDuty]
        end

        Mobile --> CDN
        Web --> CDN
        API_Client --> LB
        Mobile --> LB
        Web --> LB

        LB --> Auth
        Auth --> RateLimit
        RateLimit --> Router
        Router --> Chat_API

        Chat_API --> Conv_Service
        Chat_API --> Stream_Manager
        Chat_API --> Moderation
        Chat_API --> LLM_Gateway

        Stream_Manager --> Redis_Session
        Conv_Service --> Conv_DB
        Conv_Service --> Redis_Conv

        RateLimit --> Redis_Rate
        RateLimit --> User_DB

        LLM_Gateway --> Model_Large
        LLM_Gateway --> Model_Medium
        LLM_Gateway --> Model_Small
        LLM_Gateway --> Cache_Manager

        Cache_Manager --> Redis_Prompt

        Model_Large --> Tool_Executor
        Model_Medium --> Tool_Executor
        Model_Small --> Tool_Executor

        Tool_Executor --> Plugin_Manager

        Chat_API --> Usage_Tracker
        Usage_Tracker --> Kafka
        Usage_Tracker --> Usage_DB

        Kafka --> Message_DB
        Kafka --> Usage_DB
        Kafka --> Metrics

        Model_Store --> Model_Large
        Model_Store --> Model_Medium
        Model_Store --> Model_Small

        LLM_Gateway --> Metrics
        Chat_API --> Logs
        Metrics --> Alerts

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Session fill:#fff4e1
        style Redis_Rate fill:#fff4e1
        style Redis_Conv fill:#fff4e1
        style Redis_Prompt fill:#fff4e1
        style Conv_DB fill:#ffe1e1
        style Message_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Usage_DB fill:#e1f5e1
        style Model_Large fill:#f3e5f5
        style Model_Medium fill:#f3e5f5
        style Model_Small fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **vLLM (Inference)** | PagedAttention for KV cache, continuous batching, 2-4x throughput vs vanilla | TensorRT-LLM (less flexible), HuggingFace TGI (lower throughput), custom (too complex) |
    | **Cassandra (Messages)** | Time-series data, write-heavy (2B writes/day), horizontal scaling | PostgreSQL (write bottleneck), MongoDB (less efficient for time-series) |
    | **Redis (Multi-tier)** | Sub-millisecond latency for rate limits, sessions, caching | Memcached (no persistence), no cache (too slow for rate limiting) |
    | **WebSocket/SSE** | Bi-directional streaming, keep connection alive, push tokens as generated | HTTP polling (inefficient), gRPC (browser support limited) |
    | **PostgreSQL (Conv)** | Strong consistency for conversations, relational data, good for read-heavy | NoSQL (need transactions), DynamoDB (costly for scans) |
    | **Kafka** | Reliable event streaming, async processing, replay for analytics | RabbitMQ (lower throughput), direct calls (no buffering) |

    **Key Trade-off:** We chose **streaming responses** with WebSocket/SSE over simple request-response. This increases system complexity (connection management, backpressure) but provides much better UX with immediate feedback.

    ---

    ## API Design

    ### 1. Chat Completions (Non-streaming)

    **Request:**
    ```http
    POST /v1/chat/completions
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "model": "gpt-4",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ],
      "temperature": 0.7,
      "max_tokens": 500,
      "top_p": 1.0,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0,
      "user": "user123"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "chatcmpl-abc123",
      "object": "chat.completion",
      "created": 1738191234,
      "model": "gpt-4-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The capital of France is Paris. It is located in the north-central part of the country and is known for its art, fashion, culture, and iconic landmarks like the Eiffel Tower."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 42,
        "total_tokens": 67
      }
    }
    ```

    ### 2. Chat Completions (Streaming)

    **Request:**
    ```http
    POST /v1/chat/completions
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "model": "gpt-4",
      "messages": [
        {
          "role": "user",
          "content": "Write a short poem about AI."
        }
      ],
      "stream": true,
      "temperature": 0.8
    }
    ```

    **Response (Server-Sent Events):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: text/event-stream
    Cache-Control: no-cache
    Connection: keep-alive

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"content":"In"},"finish_reason":null}]}

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"content":" circuits"},"finish_reason":null}]}

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"content":" deep"},"finish_reason":null}]}

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"content":" and"},"finish_reason":null}]}

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{"content":" bright"},"finish_reason":null}]}

    ...

    data: {"id":"chatcmpl-xyz789","object":"chat.completion.chunk","created":1738191234,"model":"gpt-4-0125","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

    data: [DONE]
    ```

    ### 3. Create Conversation

    **Request:**
    ```http
    POST /v1/conversations
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "title": "Planning vacation to Paris",
      "metadata": {
        "tags": ["travel", "planning"],
        "archived": false
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "id": "conv_abc123",
      "object": "conversation",
      "created_at": "2026-01-29T19:30:00Z",
      "title": "Planning vacation to Paris",
      "message_count": 0,
      "metadata": {
        "tags": ["travel", "planning"],
        "archived": false
      }
    }
    ```

    ### 4. List Conversations

    **Request:**
    ```http
    GET /v1/conversations?limit=20&offset=0&order=desc
    Authorization: Bearer <api_key>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "object": "list",
      "data": [
        {
          "id": "conv_abc123",
          "title": "Planning vacation to Paris",
          "created_at": "2026-01-29T19:30:00Z",
          "updated_at": "2026-01-29T20:15:00Z",
          "message_count": 8,
          "preview": "What are the best things to do in Paris..."
        },
        {
          "id": "conv_def456",
          "title": "Python debugging help",
          "created_at": "2026-01-28T14:20:00Z",
          "updated_at": "2026-01-28T15:45:00Z",
          "message_count": 12,
          "preview": "I'm getting a TypeError in my code..."
        }
      ],
      "has_more": true,
      "total": 47
    }
    ```

    ### 5. Get Conversation Messages

    **Request:**
    ```http
    GET /v1/conversations/conv_abc123/messages?limit=50&before=msg_xyz789
    Authorization: Bearer <api_key>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "object": "list",
      "conversation_id": "conv_abc123",
      "data": [
        {
          "id": "msg_001",
          "role": "user",
          "content": "What is the capital of France?",
          "created_at": "2026-01-29T19:30:00Z",
          "tokens": 10
        },
        {
          "id": "msg_002",
          "role": "assistant",
          "content": "The capital of France is Paris...",
          "created_at": "2026-01-29T19:30:02Z",
          "tokens": 42,
          "model": "gpt-4-0125"
        }
      ],
      "has_more": false
    }
    ```

    ### 6. Function Calling (Tool Use)

    **Request:**
    ```http
    POST /v1/chat/completions
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "model": "gpt-4",
      "messages": [
        {
          "role": "user",
          "content": "What's the weather in San Francisco?"
        }
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "City name"
                },
                "unit": {
                  "type": "string",
                  "enum": ["celsius", "fahrenheit"]
                }
              },
              "required": ["location"]
            }
          }
        }
      ],
      "tool_choice": "auto"
    }
    ```

    **Response (Model requests tool call):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "chatcmpl-abc123",
      "object": "chat.completion",
      "created": 1738191234,
      "model": "gpt-4-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_xyz789",
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "arguments": "{\"location\":\"San Francisco\",\"unit\":\"fahrenheit\"}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ]
    }
    ```

    **Follow-up Request (After executing tool):**
    ```http
    POST /v1/chat/completions
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "model": "gpt-4",
      "messages": [
        {
          "role": "user",
          "content": "What's the weather in San Francisco?"
        },
        {
          "role": "assistant",
          "content": null,
          "tool_calls": [
            {
              "id": "call_xyz789",
              "type": "function",
              "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"San Francisco\",\"unit\":\"fahrenheit\"}"
              }
            }
          ]
        },
        {
          "role": "tool",
          "tool_call_id": "call_xyz789",
          "content": "{\"temperature\":58,\"condition\":\"partly cloudy\",\"humidity\":65}"
        }
      ]
    }
    ```

    **Final Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "id": "chatcmpl-def456",
      "object": "chat.completion",
      "created": 1738191240,
      "model": "gpt-4-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The current weather in San Francisco is 58°F and partly cloudy, with 65% humidity. It's a pleasant day with some clouds!"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 95,
        "completion_tokens": 32,
        "total_tokens": 127
      }
    }
    ```

=== "💾 Step 3: Database Schema"

    ## 1. User Database (PostgreSQL)

    ```sql
    -- Users and authentication
    CREATE TABLE users (
        user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        full_name VARCHAR(255),
        organization_id UUID REFERENCES organizations(org_id),
        tier VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
        status VARCHAR(50) DEFAULT 'active', -- active, suspended, deleted
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        last_login_at TIMESTAMP,
        INDEX idx_email (email),
        INDEX idx_org (organization_id)
    );

    -- Organizations for multi-tenancy
    CREATE TABLE organizations (
        org_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(255) NOT NULL,
        tier VARCHAR(50) DEFAULT 'free',
        status VARCHAR(50) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_name (name)
    );

    -- API keys for programmatic access
    CREATE TABLE api_keys (
        key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        key_prefix VARCHAR(20) NOT NULL, -- First 8 chars for display
        hashed_key TEXT NOT NULL,
        name VARCHAR(255),
        scopes JSONB DEFAULT '[]', -- ["chat:read", "chat:write"]
        last_used_at TIMESTAMP,
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_user (user_id),
        INDEX idx_key_prefix (key_prefix)
    );

    -- Rate limit quotas
    CREATE TABLE rate_limits (
        limit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
        organization_id UUID REFERENCES organizations(org_id) ON DELETE CASCADE,
        model VARCHAR(100), -- gpt-4, gpt-3.5-turbo, etc.
        tier VARCHAR(50),
        requests_per_minute INTEGER DEFAULT 60,
        requests_per_day INTEGER DEFAULT 10000,
        tokens_per_minute INTEGER DEFAULT 90000,
        tokens_per_day INTEGER DEFAULT 2000000,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_user (user_id),
        INDEX idx_org (organization_id),
        UNIQUE(user_id, model)
    );

    -- Partition by user_id range for horizontal scaling
    -- e.g., users_0, users_1, ... users_99
    ```

    ## 2. Conversation Database (PostgreSQL - Sharded by user_id)

    ```sql
    -- Conversations
    CREATE TABLE conversations (
        conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL, -- Shard key
        title VARCHAR(500),
        model VARCHAR(100), -- Primary model used
        system_prompt TEXT,
        metadata JSONB DEFAULT '{}', -- tags, custom fields
        message_count INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        archived BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_user_created (user_id, created_at DESC),
        INDEX idx_user_updated (user_id, updated_at DESC),
        INDEX idx_archived (archived)
    );

    -- Shard by user_id hash:
    -- conversations_0 (user_id % 100 = 0-9)
    -- conversations_1 (user_id % 100 = 10-19)
    -- ... conversations_9 (user_id % 100 = 90-99)

    -- Conversation sharing/permissions
    CREATE TABLE conversation_shares (
        share_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        shared_with_user_id UUID,
        shared_with_organization_id UUID,
        permission VARCHAR(50) DEFAULT 'view', -- view, comment, edit
        created_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_conversation (conversation_id),
        INDEX idx_shared_user (shared_with_user_id)
    );
    ```

    ## 3. Message Store (Cassandra - Time-series)

    ```cql
    -- Messages in a conversation
    CREATE TABLE messages (
        conversation_id UUID,
        message_id TIMEUUID,          -- Time-ordered UUID
        user_id UUID,
        role TEXT,                     -- user, assistant, system, tool
        content TEXT,
        tool_calls TEXT,               -- JSON array of tool calls
        model TEXT,                    -- Model that generated (for assistant)
        finish_reason TEXT,            -- stop, length, tool_calls, content_filter
        prompt_tokens INT,
        completion_tokens INT,
        total_tokens INT,
        latency_ms INT,
        created_at TIMESTAMP,
        metadata MAP<TEXT, TEXT>,      -- Additional key-value pairs
        PRIMARY KEY (conversation_id, message_id)
    ) WITH CLUSTERING ORDER BY (message_id DESC)
      AND compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': 1, 'compaction_window_unit': 'DAYS'}
      AND default_time_to_live = 31536000; -- 1 year retention

    -- Index for user-level queries
    CREATE TABLE messages_by_user (
        user_id UUID,
        created_at TIMESTAMP,
        conversation_id UUID,
        message_id TIMEUUID,
        PRIMARY KEY (user_id, created_at, conversation_id, message_id)
    ) WITH CLUSTERING ORDER BY (created_at DESC, conversation_id DESC, message_id DESC);

    -- Materialized view for quick message counts
    CREATE MATERIALIZED VIEW message_counts AS
        SELECT conversation_id, COUNT(*) as count
        FROM messages
        WHERE conversation_id IS NOT NULL
        GROUP BY conversation_id;
    ```

    ## 4. Usage & Billing (ClickHouse - Analytics)

    ```sql
    -- Usage logs for billing and analytics
    CREATE TABLE usage_logs (
        log_id UUID,
        timestamp DateTime64(3),
        user_id UUID,
        organization_id UUID,
        conversation_id UUID,
        message_id UUID,
        request_id String,
        model String,
        endpoint String,              -- /v1/chat/completions, etc.
        prompt_tokens UInt32,
        completion_tokens UInt32,
        total_tokens UInt32,
        latency_ms UInt32,
        ttft_ms UInt32,               -- Time to first token
        tps Float32,                  -- Tokens per second
        cached_tokens UInt32,         -- Tokens served from cache
        status_code UInt16,
        error_message String,
        ip_address IPv4,
        user_agent String,
        region String,                -- us-east-1, eu-west-1, etc.
        cost_usd Float64,             -- Calculated cost
        date Date DEFAULT toDate(timestamp)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (user_id, timestamp)
    SETTINGS index_granularity = 8192;

    -- Aggregated daily usage per user
    CREATE MATERIALIZED VIEW daily_usage_by_user
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (user_id, date, model)
    AS SELECT
        user_id,
        date,
        model,
        count() as request_count,
        sum(prompt_tokens) as total_prompt_tokens,
        sum(completion_tokens) as total_completion_tokens,
        sum(total_tokens) as total_tokens,
        avg(latency_ms) as avg_latency_ms,
        sum(cost_usd) as total_cost_usd
    FROM usage_logs
    GROUP BY user_id, date, model;
    ```

    ## 5. Cache Schema (Redis)

    ```redis
    # Session cache (10min TTL)
    Key: session:{user_id}:{session_id}
    Value: JSON {
        "user_id": "...",
        "conversation_id": "...",
        "model": "gpt-4",
        "context_tokens": 1500,
        "created_at": "..."
    }
    TTL: 600 seconds

    # Rate limit state (1min TTL, sliding window)
    Key: rate_limit:{user_id}:{model}:requests:{window}
    Value: Counter (requests in current window)
    TTL: 60 seconds

    Key: rate_limit:{user_id}:{model}:tokens:{window}
    Value: Counter (tokens in current window)
    TTL: 60 seconds

    # Token bucket (persistent)
    Key: token_bucket:{user_id}:{model}
    Value: JSON {
        "tokens": 90000,
        "last_refill": 1738191234,
        "capacity": 90000
    }
    TTL: None (persistent)

    # Conversation cache (1 hour TTL)
    Key: conversation:{conversation_id}
    Value: JSON {
        "conversation_id": "...",
        "user_id": "...",
        "title": "...",
        "messages": [...],
        "total_tokens": 2500
    }
    TTL: 3600 seconds

    # Prompt cache metadata (1 day TTL)
    Key: prompt_cache:{conversation_id}
    Value: JSON {
        "cache_key": "sha256_hash_of_prefix",
        "cached_tokens": 1500,
        "expires_at": 1738191234
    }
    TTL: 86400 seconds

    # Response cache for common queries (1 hour TTL)
    Key: response_cache:{hash_of_messages_and_params}
    Value: JSON {
        "response": "...",
        "tokens": 42,
        "model": "gpt-4",
        "created_at": 1738191234
    }
    TTL: 3600 seconds

    # Active streaming connections
    Key: stream:{request_id}
    Value: JSON {
        "user_id": "...",
        "conversation_id": "...",
        "model": "gpt-4",
        "started_at": 1738191234,
        "tokens_streamed": 25
    }
    TTL: 300 seconds
    ```

=== "🔍 Step 4: Deep Dive - Critical Components"

    ## 1. LLM Inference Serving (vLLM)

    ### Why vLLM?

    - **PagedAttention:** Efficient KV cache management (no fragmentation)
    - **Continuous batching:** Add new requests to existing batch dynamically
    - **Higher throughput:** 2-4x better than vanilla HuggingFace
    - **Prefix caching:** Reuse KV cache for common prompt prefixes

    ### vLLM Architecture

    ```python
    # vLLM server configuration
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    # Initialize engine
    engine_args = AsyncEngineArgs(
        model="gpt-4-175b",
        tensor_parallel_size=8,        # 8 GPUs for model parallelism
        max_num_batched_tokens=32768,  # Max tokens in batch
        max_num_seqs=256,               # Max concurrent sequences
        gpu_memory_utilization=0.90,   # Use 90% of GPU memory
        enable_prefix_caching=True,    # Enable prompt caching
        max_model_len=8192,            # Max context window
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Request processing
    async def generate_streaming(
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams
    ):
        """Stream tokens as they're generated"""
        results_generator = engine.generate(
            prompt,
            sampling_params,
            request_id
        )

        async for request_output in results_generator:
            # Each output contains newly generated tokens
            text = request_output.outputs[0].text
            yield text
    ```

    ### Continuous Batching

    ```
    Traditional Static Batching:
    Batch 1: [Req1, Req2, Req3, Req4] → Wait for ALL to complete
    Batch 2: [Req5, Req6, Req7, Req8] → Start after Batch 1 done

    Problem: Long requests block short ones
    GPU Utilization: ~50% (idle during padding)

    vLLM Continuous Batching:
    Time 0: [Req1, Req2, Req3, Req4]
    Time 1: [Req1, Req2, Req5, Req6]  ← Req3,4 done, add Req5,6
    Time 2: [Req1, Req5, Req6, Req7]  ← Req2 done, add Req7

    Benefit: No idle GPUs, higher throughput
    GPU Utilization: ~85-90%
    ```

    ### Prefix Caching (Prompt Caching)

    ```python
    # Example: System prompt is shared across requests
    system_prompt = "You are a helpful AI assistant..."  # 500 tokens

    # Request 1
    messages_1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is Python?"}
    ]
    # KV cache for system_prompt: COMPUTE and CACHE
    # Cost: 500 tokens compute

    # Request 2 (same system prompt)
    messages_2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is JavaScript?"}
    ]
    # KV cache for system_prompt: REUSE from cache
    # Cost: 0 tokens compute (saved 500 tokens!)

    # Savings for conversation context
    conversation_context = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]  # 3000 tokens

    # Next turn reuses all previous context
    # Only compute new user message + generate response
    # Saves 3000 tokens of compute per turn
    ```

    ### GPU Scaling Math

    ```
    Single A100 (80GB) throughput with vLLM:
    - Model: GPT-4 class (175B params, FP16)
    - Model weights: 350 GB → Need 8x A100 (tensor parallelism)
    - Effective memory per node: 640 GB total, 350 GB for model
    - Remaining: 290 GB for KV cache

    KV cache per request:
    - Context: 8K tokens, FP16, 48 layers
    - Cache size: 8K × 2 (key+value) × 2 bytes × 48 = ~1.5 MB per token
    - 8K tokens: ~12 MB per request

    Batch size:
    - Available memory: 290 GB
    - Memory per request: 12 MB
    - Max concurrent: 290,000 MB / 12 MB = ~24,000 requests
    - Practical batch size: 64-256 (due to compute limits)

    Throughput:
    - Tokens per second per GPU node: ~2,000 tokens/sec
    - With continuous batching: ~2,500 tokens/sec
    - Required nodes for 9.2M tokens/sec: 9,200,000 / 2,500 = 3,680 GPU nodes
    - Each node = 8x A100 GPUs
    - Total GPUs: 3,680 × 8 = 29,440 A100 GPUs

    Cost optimization with prefix caching:
    - 50% of tokens are repeated context
    - Effective throughput: 2,500 × 1.5 = 3,750 tokens/sec per node
    - Required nodes: 9,200,000 / 3,750 = 2,453 nodes
    - Total GPUs: 2,453 × 8 = 19,624 A100 GPUs (33% savings!)
    ```

    ---

    ## 2. Streaming Implementation (SSE)

    ### Server-Sent Events (SSE)

    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from vllm import AsyncLLMEngine, SamplingParams
    import json
    import asyncio

    app = FastAPI()

    async def generate_stream(
        engine: AsyncLLMEngine,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams
    ):
        """Stream tokens using SSE format"""
        # Send initial chunk with role
        initial = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(initial)}\n\n"

        # Stream generated tokens
        previous_text = ""
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            new_text = request_output.outputs[0].text
            delta_text = new_text[len(previous_text):]
            previous_text = new_text

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # Check if generation is complete
            if request_output.finished:
                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        data = await request.json()

        if data.get("stream"):
            # Streaming response
            return StreamingResponse(
                generate_stream(
                    engine=llm_engine,
                    prompt=format_prompt(data["messages"]),
                    request_id=generate_id(),
                    sampling_params=SamplingParams(
                        temperature=data.get("temperature", 0.7),
                        max_tokens=data.get("max_tokens", 500)
                    )
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        else:
            # Non-streaming response
            # ... regular response handling
    ```

    ### Client-side SSE Handling

    ```javascript
    // JavaScript client for SSE streaming
    async function streamChatCompletion(messages) {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4',
                messages: messages,
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        console.log('Stream complete');
                        return;
                    }

                    try {
                        const chunk = JSON.parse(data);
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            // Append to UI
                            appendToMessage(content);
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
        }
    }

    // Usage
    streamChatCompletion([
        { role: 'user', content: 'Write a poem about AI' }
    ]);
    ```

    ### Backpressure Handling

    ```python
    import asyncio
    from asyncio import Queue

    class StreamManager:
        def __init__(self, max_buffer_size: int = 100):
            self.max_buffer_size = max_buffer_size
            self.active_streams = {}

        async def stream_with_backpressure(
            self,
            request_id: str,
            token_generator
        ):
            """Handle backpressure if client is slow"""
            buffer = Queue(maxsize=self.max_buffer_size)

            async def producer():
                """Produce tokens from LLM"""
                try:
                    async for token in token_generator:
                        await buffer.put(token)
                except Exception as e:
                    await buffer.put({"error": str(e)})
                finally:
                    await buffer.put(None)  # Signal completion

            async def consumer():
                """Consume tokens and yield to client"""
                while True:
                    token = await buffer.get()
                    if token is None:
                        break
                    if isinstance(token, dict) and "error" in token:
                        raise Exception(token["error"])
                    yield token

            # Start producer in background
            producer_task = asyncio.create_task(producer())

            try:
                async for token in consumer():
                    yield token
            finally:
                # Cleanup
                producer_task.cancel()
                del self.active_streams[request_id]
    ```

    ---

    ## 3. Context Window Management

    ### Sliding Window for Long Conversations

    ```python
    from typing import List, Dict

    class ContextManager:
        def __init__(self, max_tokens: int = 8192):
            self.max_tokens = max_tokens
            self.system_tokens = 0
            self.reserved_output_tokens = 500

        def truncate_messages(
            self,
            messages: List[Dict],
            model: str
        ) -> List[Dict]:
            """
            Keep most recent messages that fit in context window.
            Always keep system message.
            """
            # Tokenize all messages
            tokenized = []
            for msg in messages:
                tokens = self.count_tokens(msg["content"], model)
                tokenized.append({
                    "message": msg,
                    "tokens": tokens
                })

            # Always keep system message
            system_msgs = [t for t in tokenized if t["message"]["role"] == "system"]
            other_msgs = [t for t in tokenized if t["message"]["role"] != "system"]

            system_tokens = sum(t["tokens"] for t in system_msgs)
            available_tokens = self.max_tokens - system_tokens - self.reserved_output_tokens

            # Keep most recent messages
            selected = []
            total_tokens = 0
            for t in reversed(other_msgs):
                if total_tokens + t["tokens"] <= available_tokens:
                    selected.insert(0, t)
                    total_tokens += t["tokens"]
                else:
                    break

            # Combine system + selected messages
            result = [t["message"] for t in system_msgs] + [t["message"] for t in selected]

            return result

        def count_tokens(self, text: str, model: str) -> int:
            """Count tokens using tiktoken"""
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))

    # Usage
    manager = ContextManager(max_tokens=8192)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Paris."},
        {"role": "assistant", "content": "Paris is the capital of France..."},
        {"role": "user", "content": "What about Rome?"},
        {"role": "assistant", "content": "Rome is the capital of Italy..."},
        # ... 50 more messages
        {"role": "user", "content": "Summarize our conversation."}
    ]

    # Truncate to fit in context window
    truncated = manager.truncate_messages(messages, model="gpt-4")
    # Result: System message + last N messages that fit
    ```

    ### Summarization Strategy

    ```python
    class ConversationSummarizer:
        """Summarize old messages to save context"""

        async def summarize_and_compact(
            self,
            messages: List[Dict],
            model: str,
            target_tokens: int
        ) -> List[Dict]:
            """
            If conversation too long:
            1. Keep system message
            2. Keep last N messages (most recent)
            3. Summarize middle messages
            """
            if self.count_tokens(messages) <= target_tokens:
                return messages

            # Keep system and last 6 messages
            system = [m for m in messages if m["role"] == "system"]
            recent = messages[-6:]
            middle = messages[len(system):-6]

            # Summarize middle
            summary_prompt = f"""Summarize this conversation history concisely:

{self.format_messages(middle)}

Provide a brief summary of the key points discussed."""

            summary = await self.llm_generate(summary_prompt, model)

            # Reconstruct
            return system + [
                {"role": "system", "content": f"Previous conversation summary: {summary}"}
            ] + recent
    ```

    ---

    ## 4. Rate Limiting (Token Bucket Algorithm)

    ### Token Bucket Implementation

    ```python
    import time
    from typing import Optional
    import redis

    class TokenBucketRateLimiter:
        def __init__(self, redis_client: redis.Redis):
            self.redis = redis_client

        async def check_rate_limit(
            self,
            user_id: str,
            model: str,
            tokens_needed: int,
            rpm_limit: int = 60,
            tpm_limit: int = 90000
        ) -> tuple[bool, Optional[str]]:
            """
            Check if request is within rate limits.
            Returns (allowed, error_message)
            """
            now = int(time.time())

            # Request rate limiting (RPM)
            rpm_key = f"rate_limit:{user_id}:{model}:rpm"
            requests_this_minute = await self.redis.incr(rpm_key)

            if requests_this_minute == 1:
                # First request in this minute, set expiry
                await self.redis.expire(rpm_key, 60)

            if requests_this_minute > rpm_limit:
                return False, f"Rate limit exceeded: {rpm_limit} requests per minute"

            # Token rate limiting (TPM) - Token bucket
            tpm_key = f"rate_limit:{user_id}:{model}:tokens"

            # Get current bucket state
            bucket = await self.redis.get(tpm_key)
            if bucket:
                bucket_data = json.loads(bucket)
                available_tokens = bucket_data["tokens"]
                last_refill = bucket_data["last_refill"]
            else:
                available_tokens = tpm_limit
                last_refill = now

            # Refill tokens based on time elapsed
            time_elapsed = now - last_refill
            refill_rate = tpm_limit / 60  # Tokens per second
            tokens_to_add = int(time_elapsed * refill_rate)
            available_tokens = min(tpm_limit, available_tokens + tokens_to_add)

            # Check if enough tokens
            if available_tokens < tokens_needed:
                retry_after = int((tokens_needed - available_tokens) / refill_rate)
                return False, f"Token quota exceeded. Retry after {retry_after}s"

            # Deduct tokens
            available_tokens -= tokens_needed

            # Update bucket
            bucket_data = {
                "tokens": available_tokens,
                "last_refill": now
            }
            await self.redis.set(tpm_key, json.dumps(bucket_data), ex=120)

            return True, None

    # Usage in API
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Estimate tokens needed
        input_tokens = count_tokens(request.messages)
        estimated_output_tokens = request.max_tokens or 500
        total_tokens = input_tokens + estimated_output_tokens

        # Check rate limit
        allowed, error = await rate_limiter.check_rate_limit(
            user_id=request.user_id,
            model=request.model,
            tokens_needed=total_tokens
        )

        if not allowed:
            raise HTTPException(status_code=429, detail=error)

        # Proceed with request
        # ...
    ```

    ### Tiered Rate Limiting

    ```python
    RATE_LIMITS = {
        "free": {
            "gpt-4": {
                "rpm": 3,
                "tpm": 40000,
                "rpd": 200
            },
            "gpt-3.5-turbo": {
                "rpm": 60,
                "tpm": 90000,
                "rpd": 10000
            }
        },
        "pro": {
            "gpt-4": {
                "rpm": 60,
                "tpm": 90000,
                "rpd": 10000
            },
            "gpt-3.5-turbo": {
                "rpm": 3500,
                "tpm": 1000000,
                "rpd": 500000
            }
        },
        "enterprise": {
            "gpt-4": {
                "rpm": 10000,
                "tpm": 10000000,
                "rpd": 1000000000
            },
            "gpt-3.5-turbo": {
                "rpm": 10000,
                "tpm": 10000000,
                "rpd": 1000000000
            }
        }
    }

    async def get_rate_limit(user_id: str, model: str):
        """Get rate limit for user tier"""
        user = await get_user(user_id)
        tier = user.tier
        return RATE_LIMITS[tier][model]
    ```

    ---

    ## 5. Conversation State Management

    ### In-Memory Context Cache

    ```python
    from typing import List, Dict, Optional
    import hashlib
    import json

    class ConversationCache:
        """Cache conversation context for faster repeated access"""

        def __init__(self, redis_client: redis.Redis):
            self.redis = redis_client
            self.ttl = 3600  # 1 hour

        async def get_conversation(
            self,
            conversation_id: str
        ) -> Optional[Dict]:
            """Get conversation from cache"""
            key = f"conversation:{conversation_id}"
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None

        async def set_conversation(
            self,
            conversation_id: str,
            messages: List[Dict],
            metadata: Dict
        ):
            """Cache conversation"""
            key = f"conversation:{conversation_id}"
            data = {
                "messages": messages,
                "metadata": metadata,
                "cached_at": int(time.time())
            }
            await self.redis.set(key, json.dumps(data), ex=self.ttl)

        async def append_message(
            self,
            conversation_id: str,
            message: Dict
        ):
            """Append message to cached conversation"""
            conversation = await self.get_conversation(conversation_id)
            if conversation:
                conversation["messages"].append(message)
                await self.set_conversation(
                    conversation_id,
                    conversation["messages"],
                    conversation["metadata"]
                )

        async def get_prompt_cache_key(
            self,
            messages: List[Dict]
        ) -> str:
            """Generate cache key for prompt prefix"""
            # Hash the messages to create cache key
            content = json.dumps(messages, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()

        async def check_prefix_cache(
            self,
            messages: List[Dict]
        ) -> Optional[Dict]:
            """Check if prompt prefix is cached (for KV cache reuse)"""
            cache_key = await self.get_prompt_cache_key(messages[:-1])  # All but last
            key = f"prompt_cache:{cache_key}"
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
    ```

    ---

    ## 6. Content Moderation

    ```python
    from typing import List, Dict, Tuple

    class ContentModerator:
        """Filter harmful/unsafe content"""

        def __init__(self):
            self.blocked_patterns = self.load_blocked_patterns()
            self.moderation_model = self.load_moderation_model()

        async def moderate_input(
            self,
            messages: List[Dict]
        ) -> Tuple[bool, Optional[str]]:
            """
            Check if input contains harmful content.
            Returns (is_safe, reason)
            """
            for message in messages:
                if message["role"] == "user":
                    content = message["content"]

                    # Pattern matching for obvious violations
                    if self.contains_blocked_pattern(content):
                        return False, "Content violates usage policies"

                    # ML-based moderation
                    categories = await self.classify_content(content)
                    if self.is_violating(categories):
                        return False, f"Flagged for: {', '.join(categories)}"

            return True, None

        async def moderate_output(
            self,
            output: str
        ) -> Tuple[bool, Optional[str]]:
            """
            Check if model output contains harmful content.
            Returns (is_safe, reason)
            """
            categories = await self.classify_content(output)
            if self.is_violating(categories):
                return False, f"Response flagged for: {', '.join(categories)}"
            return True, None

        async def classify_content(self, text: str) -> Dict[str, float]:
            """
            Classify content into categories with scores.
            Categories: hate, violence, sexual, self-harm, etc.
            """
            # Use OpenAI moderation API or custom model
            result = await self.moderation_model.classify(text)
            return result  # {"hate": 0.1, "violence": 0.05, ...}

        def is_violating(self, categories: Dict[str, float]) -> bool:
            """Check if any category exceeds threshold"""
            thresholds = {
                "hate": 0.8,
                "violence": 0.8,
                "sexual": 0.9,
                "self-harm": 0.7
            }
            for category, score in categories.items():
                if score > thresholds.get(category, 0.9):
                    return True
            return False

    # Usage in API
    moderator = ContentModerator()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Moderate input
        is_safe, reason = await moderator.moderate_input(request.messages)
        if not is_safe:
            raise HTTPException(status_code=400, detail=reason)

        # Generate response
        response = await generate_response(request)

        # Moderate output
        is_safe, reason = await moderator.moderate_output(response)
        if not is_safe:
            # Log incident and return safe error message
            await log_moderation_event(request, response, reason)
            raise HTTPException(
                status_code=500,
                detail="Unable to generate appropriate response"
            )

        return response
    ```

=== "📈 Step 5: Scalability & Optimization"

    ## Scalability Considerations

    ### 1. Horizontal Scaling

    **API Layer:**
    ```
    - Stateless API servers (easy to scale)
    - Load balancer with sticky sessions for streaming
    - Auto-scaling based on request queue depth
    - Target: 70% CPU utilization
    - Scale out when queue > 100 requests
    ```

    **GPU Inference Layer:**
    ```
    - Model replicas across multiple GPU nodes
    - Request routing based on model type
    - Dynamic batching to maximize GPU utilization
    - Health checks and automatic failover
    - Gradual rollout for model updates
    ```

    **Database Layer:**
    ```
    - PostgreSQL: Shard by user_id (100 shards)
    - Cassandra: Partition by conversation_id
    - Redis: Cluster mode with 10 nodes
    - Read replicas for conversation history
    ```

    ### 2. GPU Auto-scaling

    ```python
    class GPUAutoscaler:
        """Auto-scale GPU inference nodes based on load"""

        def __init__(self):
            self.min_nodes = 10
            self.max_nodes = 1000
            self.target_utilization = 0.75
            self.scale_up_threshold = 0.85
            self.scale_down_threshold = 0.60

        async def check_and_scale(self):
            """Check metrics and scale if needed"""
            metrics = await self.get_gpu_metrics()

            avg_utilization = metrics["avg_gpu_utilization"]
            queue_depth = metrics["request_queue_depth"]
            current_nodes = metrics["active_nodes"]

            # Scale up if high utilization or queue building
            if (avg_utilization > self.scale_up_threshold or
                queue_depth > 1000):
                new_nodes = min(
                    current_nodes + 10,
                    self.max_nodes
                )
                await self.scale_to(new_nodes)
                await self.alert(f"Scaling up to {new_nodes} nodes")

            # Scale down if low utilization
            elif (avg_utilization < self.scale_down_threshold and
                  queue_depth < 100):
                new_nodes = max(
                    current_nodes - 5,
                    self.min_nodes
                )
                await self.scale_to(new_nodes)
                await self.alert(f"Scaling down to {new_nodes} nodes")

        async def scale_to(self, target_nodes: int):
            """Scale GPU cluster to target nodes"""
            # Gradual scaling to avoid thundering herd
            current = await self.get_active_nodes()

            if target_nodes > current:
                # Scale up: Add nodes incrementally
                for i in range(target_nodes - current):
                    await self.add_gpu_node()
                    await asyncio.sleep(30)  # Wait for warmup
            elif target_nodes < current:
                # Scale down: Remove nodes gracefully
                for i in range(current - target_nodes):
                    await self.remove_gpu_node_gracefully()
                    await asyncio.sleep(10)

        async def remove_gpu_node_gracefully(self):
            """Drain node before removing"""
            # 1. Mark node as draining (stop accepting new requests)
            # 2. Wait for active requests to complete
            # 3. Remove node from pool
            pass
    ```

    ### 3. Cost Optimization

    **Model Routing:**
    ```python
    class IntelligentRouter:
        """Route requests to appropriate model based on complexity"""

        async def route_request(self, messages: List[Dict]) -> str:
            """
            Analyze request and route to appropriate model:
            - Simple queries → Small model (cheap)
            - Complex reasoning → Large model (expensive)
            """
            complexity_score = await self.analyze_complexity(messages)

            if complexity_score < 0.3:
                return "gpt-3.5-turbo"  # Fast, cheap
            elif complexity_score < 0.7:
                return "gpt-4-mini"  # Balanced
            else:
                return "gpt-4"  # Most capable

        async def analyze_complexity(self, messages: List[Dict]) -> float:
            """
            Score query complexity (0-1):
            - Keywords: "explain", "analyze", "compare" → High
            - Short queries → Low
            - Code generation → High
            """
            last_message = messages[-1]["content"].lower()

            complexity_keywords = [
                "analyze", "compare", "explain", "reason",
                "debug", "optimize", "design", "architecture"
            ]

            score = 0.0

            # Length-based
            if len(last_message) > 200:
                score += 0.3

            # Keyword-based
            for keyword in complexity_keywords:
                if keyword in last_message:
                    score += 0.2

            # Code detection
            if "```" in last_message or "def " in last_message:
                score += 0.3

            return min(score, 1.0)
    ```

    **Prompt Caching Savings:**
    ```
    Without caching:
    - 2B requests/day × 2,000 tokens/request = 4T tokens/day
    - Cost: $4T × $0.03/1M = $120,000/day

    With 50% cache hit rate:
    - Cached: 2T tokens (free)
    - Computed: 2T tokens × $0.03/1M = $60,000/day
    - Savings: $60,000/day = $1.8M/month = $21.6M/year
    ```

    **Spot Instance Strategy:**
    ```
    GPU cluster composition:
    - 70% on-demand instances (guaranteed capacity)
    - 30% spot instances (70% cost savings)

    Spot instance handling:
    - Use for non-critical workloads
    - Graceful migration when spot interrupted
    - Maintain minimum capacity with on-demand

    Cost savings:
    - On-demand: 1,680 GPUs × $2.50/hr = $4,200/hr
    - Spot: 720 GPUs × $0.75/hr = $540/hr
    - Total: $4,740/hr (vs $6,000/hr all on-demand)
    - Savings: 21% = $9M/year
    ```

    ### 4. Caching Strategy

    ```
    Multi-layer caching:

    L1: Response cache (Redis)
    - Cache exact responses for common queries
    - Hit rate: 15-20%
    - TTL: 1 hour
    - Savings: Avoid GPU entirely

    L2: Prompt prefix cache (vLLM KV cache)
    - Cache KV for common prompt prefixes
    - Hit rate: 40-60%
    - TTL: 24 hours
    - Savings: Reduce compute by 50%

    L3: Conversation cache (Redis)
    - Cache full conversation context
    - Hit rate: 80%
    - TTL: 1 hour
    - Savings: Avoid DB queries

    Total cache hit improvement:
    - L1: 20% requests skip GPU → 20% cost saved
    - L2: 50% of remaining use cached KV → 40% cost saved
    - L3: 80% avoid DB → 80% latency reduction
    - Combined: ~50% cost reduction, 60% latency improvement
    ```

    ### 5. Database Sharding

    **User/Conversation Sharding:**
    ```python
    class DatabaseShardRouter:
        """Route queries to correct database shard"""

        def __init__(self, num_shards: int = 100):
            self.num_shards = num_shards
            self.shards = self.initialize_shards()

        def get_shard(self, user_id: str) -> Database:
            """Get shard for user_id"""
            shard_id = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % self.num_shards
            return self.shards[shard_id]

        async def get_conversations(self, user_id: str):
            """Get all conversations for user"""
            shard = self.get_shard(user_id)
            return await shard.query(
                "SELECT * FROM conversations WHERE user_id = $1",
                user_id
            )

        async def get_conversation_by_id(self, conversation_id: str):
            """
            Get conversation by ID.
            Need to query shard mapping table first.
            """
            # Option 1: Store shard info in conversation_id
            # Format: {shard_id}_{uuid}
            shard_id = int(conversation_id.split("_")[0])
            shard = self.shards[shard_id]

            return await shard.query(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id
            )

    # Sharding strategy
    # Shard by user_id for conversations (user queries)
    # Store conversation_id → user_id mapping in Redis for direct lookups
    ```

    ### 6. Load Balancing

    ```
    API Layer:
    - L7 load balancer (nginx/Envoy)
    - Sticky sessions for streaming (consistent hash by connection_id)
    - Health checks: /health endpoint
    - Circuit breaker: Fail fast if downstream unhealthy

    GPU Inference Layer:
    - Least-outstanding-requests (LOR) algorithm
    - Track pending requests per GPU node
    - Route new requests to least loaded node
    - Periodic re-balancing

    Algorithm:
    1. Client → API Gateway
    2. API Gateway → Rate Limiter
    3. Rate Limiter → Request Router
    4. Router queries GPU node metrics
    5. Route to node with lowest queue depth
    6. Stream response back through same path
    ```

=== "💡 Step 6: Trade-offs & Alternatives"

    ## Key Trade-offs

    ### 1. Streaming vs Batch Responses

    **Streaming (Chosen):**
    - ✅ Lower perceived latency (TTFT < 500ms)
    - ✅ Better UX (see tokens as generated)
    - ✅ Can cancel early if response not useful
    - ❌ More complex (connection management, backpressure)
    - ❌ Higher resource usage (open connections)
    - ❌ Harder to cache

    **Batch:**
    - ✅ Simpler implementation
    - ✅ Easier to cache full responses
    - ✅ Better for high-throughput batch processing
    - ❌ Higher perceived latency (wait for full response)
    - ❌ Worse UX for long responses

    **Decision:** Streaming for interactive chat, batch for API/background jobs

    ---

    ### 2. Model Serving: vLLM vs TensorRT-LLM vs Custom

    **vLLM (Chosen):**
    - ✅ Highest throughput (2-4x vs baseline)
    - ✅ PagedAttention (efficient KV cache)
    - ✅ Continuous batching
    - ✅ Easy to use, good community support
    - ❌ Python-based (some overhead)

    **TensorRT-LLM:**
    - ✅ Lower latency per request
    - ✅ Optimized for NVIDIA hardware
    - ✅ C++ based (lower overhead)
    - ❌ Less flexible batching
    - ❌ Harder to customize

    **Custom Solution:**
    - ✅ Full control over optimizations
    - ❌ Months of development time
    - ❌ Hard to maintain
    - ❌ Likely lower performance than specialized tools

    **Decision:** vLLM for production (best throughput/ease-of-use tradeoff)

    ---

    ### 3. Context Management: Sliding Window vs Summarization

    **Sliding Window (Chosen for most):**
    - ✅ Simple, fast
    - ✅ No additional LLM calls
    - ✅ Preserves recent context exactly
    - ❌ Loses older context
    - ❌ May lose important early information

    **Summarization:**
    - ✅ Retains key information from entire conversation
    - ✅ More context-aware
    - ❌ Additional LLM call (latency + cost)
    - ❌ Summarization may lose nuance
    - ❌ More complex implementation

    **Decision:** Sliding window for general use, summarization for long conversations (>50 turns)

    ---

    ### 4. Rate Limiting: User-level vs Organization-level

    **Hybrid (Chosen):**
    ```
    Rate limits at multiple levels:
    1. User-level: 60 RPM, 90K TPM
    2. Organization-level: 1000 RPM, 1M TPM
    3. API key-level: Custom limits per key

    Check all levels, fail if any exceeded
    ```

    **User-level only:**
    - ✅ Simple to implement
    - ❌ Organization can't pool quota

    **Organization-level only:**
    - ✅ Flexible quota sharing
    - ❌ Single user can consume all quota

    ---

    ### 5. Storage: PostgreSQL vs DynamoDB

    **PostgreSQL (Chosen for conversations):**
    - ✅ Strong consistency
    - ✅ Rich querying (JOIN, aggregate)
    - ✅ ACID transactions
    - ✅ Cost-effective at scale with sharding
    - ❌ Requires manual sharding
    - ❌ More operational overhead

    **DynamoDB:**
    - ✅ Fully managed, auto-scaling
    - ✅ Low latency (single-digit ms)
    - ❌ Expensive for large datasets
    - ❌ Limited querying (no JOIN)

    **Decision:** PostgreSQL for structured data (users, conversations), Cassandra for time-series (messages), DynamoDB for low-latency lookups (feature flags)

    ---

    ### 6. Multi-tenancy: Separate DBs vs Shared DB

    **Shared DB with tenant_id (Chosen):**
    - ✅ Cost-effective (fewer databases)
    - ✅ Easier to maintain
    - ✅ Better resource utilization
    - ❌ Risk of data leakage (must be careful)
    - ❌ Noisy neighbor problem

    **Separate DB per tenant:**
    - ✅ Complete isolation
    - ✅ Easier to guarantee SLA per tenant
    - ❌ Expensive (many databases)
    - ❌ Complex management

    **Decision:** Shared DB for small/medium customers, dedicated DB for enterprise

    ---

    ## Alternative Approaches

    ### 1. Monolithic vs Microservices

    **Microservices (Chosen):**
    ```
    Services:
    - API Gateway
    - Auth Service
    - Chat Service
    - Inference Service
    - Billing Service
    - Moderation Service

    Benefits:
    - Independent scaling
    - Technology flexibility
    - Fault isolation

    Drawbacks:
    - Network overhead
    - Complex deployment
    - Distributed tracing needed
    ```

    **Monolithic:**
    ```
    Single application:
    - All features in one codebase
    - Simpler deployment
    - Lower latency (no network calls)
    - Harder to scale different components independently
    ```

    ---

    ### 2. Synchronous vs Asynchronous Processing

    **Hybrid (Chosen):**
    ```
    Synchronous:
    - Chat completions (real-time requirement)
    - Rate limiting checks
    - Authentication

    Asynchronous:
    - Usage logging (via Kafka)
    - Billing calculations
    - Analytics
    - Model performance tracking

    Benefit: Low latency for user-facing, high throughput for background
    ```

    ---

    ### 3. Edge Deployment

    **Centralized GPU clusters (Chosen):**
    - ✅ Better GPU utilization via batching
    - ✅ Easier to manage/update models
    - ❌ Higher latency for distant users

    **Edge deployment:**
    - ✅ Lower latency (closer to users)
    - ❌ Harder to achieve high GPU utilization
    - ❌ Complex model distribution
    - ❌ Much higher cost (more GPUs needed)

    **Future:** Edge deployment for smaller models, centralized for large models

=== "🎯 Step 7: Extended Features"

    ## 1. Plugin/Tool Calling System

    ### Tool Definition

    ```python
    from typing import List, Dict, Callable, Any
    from pydantic import BaseModel

    class ToolParameter(BaseModel):
        name: str
        type: str
        description: str
        required: bool = False
        enum: List[str] = None

    class Tool(BaseModel):
        name: str
        description: str
        parameters: List[ToolParameter]
        function: Callable

    # Example tools
    async def get_weather(location: str, unit: str = "fahrenheit") -> Dict:
        """Get current weather for location"""
        # Call weather API
        response = await weather_api.get_current(location, unit)
        return response

    async def search_web(query: str, num_results: int = 5) -> List[Dict]:
        """Search the web for information"""
        results = await search_engine.search(query, limit=num_results)
        return results

    async def calculate(expression: str) -> float:
        """Evaluate mathematical expression"""
        # Safe eval with restricted scope
        return safe_eval(expression)

    # Register tools
    tools = [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters=[
                ToolParameter(name="location", type="string", description="City name", required=True),
                ToolParameter(name="unit", type="string", description="Temperature unit", enum=["celsius", "fahrenheit"])
            ],
            function=get_weather
        ),
        Tool(
            name="search_web",
            description="Search the internet for information",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query", required=True),
                ToolParameter(name="num_results", type="integer", description="Number of results")
            ],
            function=search_web
        )
    ]
    ```

    ### Tool Execution Flow

    ```python
    class ToolExecutor:
        """Execute tools called by LLM"""

        def __init__(self, tools: List[Tool]):
            self.tools = {tool.name: tool for tool in tools}

        async def execute_tool_calls(
            self,
            tool_calls: List[Dict]
        ) -> List[Dict]:
            """Execute all tool calls and return results"""
            results = []

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                # Get tool
                tool = self.tools.get(tool_name)
                if not tool:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": f"Error: Tool {tool_name} not found"
                    })
                    continue

                # Execute tool
                try:
                    result = await tool.function(**arguments)
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": f"Error: {str(e)}"
                    })

            return results

    # Usage in chat flow
    async def chat_with_tools(messages: List[Dict], tools: List[Tool], max_iterations: int = 5):
        """Chat loop with tool calling"""
        executor = ToolExecutor(tools)
        iteration = 0

        while iteration < max_iterations:
            # Generate response
            response = await llm.generate(messages, tools=tools)

            # Check if tool calls requested
            if response.finish_reason == "tool_calls":
                # Execute tools
                tool_results = await executor.execute_tool_calls(response.tool_calls)

                # Add assistant response and tool results to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response.tool_calls
                })
                messages.extend(tool_results)

                # Continue loop to generate final response
                iteration += 1
            else:
                # No more tool calls, return final response
                return response

        # Max iterations reached
        return {"error": "Max tool iterations reached"}
    ```

    ---

    ## 2. Usage Tracking & Billing

    ```python
    class UsageTracker:
        """Track token usage for billing"""

        async def log_usage(
            self,
            user_id: str,
            organization_id: str,
            conversation_id: str,
            message_id: str,
            model: str,
            prompt_tokens: int,
            completion_tokens: int,
            latency_ms: int,
            ttft_ms: int
        ):
            """Log usage event to Kafka"""
            event = {
                "log_id": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "user_id": user_id,
                "organization_id": organization_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "cost_usd": self.calculate_cost(model, prompt_tokens, completion_tokens)
            }

            await self.kafka_producer.send("usage_logs", event)

        def calculate_cost(
            self,
            model: str,
            prompt_tokens: int,
            completion_tokens: int
        ) -> float:
            """Calculate cost based on token usage"""
            pricing = {
                "gpt-4": {
                    "prompt": 0.03 / 1000,      # $0.03 per 1K tokens
                    "completion": 0.06 / 1000   # $0.06 per 1K tokens
                },
                "gpt-3.5-turbo": {
                    "prompt": 0.0015 / 1000,
                    "completion": 0.002 / 1000
                }
            }

            rates = pricing.get(model, pricing["gpt-3.5-turbo"])
            cost = (prompt_tokens * rates["prompt"] +
                    completion_tokens * rates["completion"])
            return round(cost, 6)

    # Billing calculation (daily job)
    async def calculate_daily_bill(organization_id: str, date: str):
        """Calculate bill for organization for given date"""
        query = f"""
        SELECT
            model,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost
        FROM usage_logs
        WHERE organization_id = '{organization_id}'
          AND date = '{date}'
        GROUP BY model
        """

        results = await clickhouse.query(query)

        total_cost = sum(r["total_cost"] for r in results)

        # Create invoice
        invoice = {
            "organization_id": organization_id,
            "date": date,
            "line_items": results,
            "total_cost": total_cost
        }

        await billing_db.save_invoice(invoice)
        return invoice
    ```

    ---

    ## 3. A/B Testing Framework

    ```python
    class ABTestManager:
        """Manage A/B tests for model parameters, features"""

        async def get_experiment_assignment(
            self,
            user_id: str,
            experiment_name: str
        ) -> str:
            """Get user's experiment variant"""
            # Check if user already assigned
            cache_key = f"experiment:{experiment_name}:{user_id}"
            variant = await self.redis.get(cache_key)

            if variant:
                return variant

            # Get experiment config
            experiment = await self.get_experiment(experiment_name)
            if not experiment or not experiment["active"]:
                return "control"

            # Assign variant based on hash
            hash_val = int(hashlib.md5(f"{user_id}{experiment_name}".encode()).hexdigest(), 16)
            rand = hash_val % 100

            cumulative = 0
            for variant_name, percentage in experiment["variants"].items():
                cumulative += percentage
                if rand < cumulative:
                    # Cache assignment
                    await self.redis.set(cache_key, variant_name, ex=86400)
                    return variant_name

            return "control"

        async def apply_experiment(
            self,
            user_id: str,
            base_params: Dict
        ) -> Dict:
            """Apply active experiments to parameters"""
            # Check active experiments
            experiments = await self.get_active_experiments()

            params = base_params.copy()
            for exp in experiments:
                variant = await self.get_experiment_assignment(user_id, exp["name"])
                if variant != "control":
                    # Apply variant parameters
                    params.update(exp["variants"][variant]["params"])

            return params

    # Example experiments
    experiments = [
        {
            "name": "temperature_test",
            "active": True,
            "variants": {
                "control": {"percentage": 50, "params": {"temperature": 0.7}},
                "variant_a": {"percentage": 25, "params": {"temperature": 0.5}},
                "variant_b": {"percentage": 25, "params": {"temperature": 0.9}}
            }
        },
        {
            "name": "max_tokens_test",
            "active": True,
            "variants": {
                "control": {"percentage": 50, "params": {"max_tokens": 500}},
                "variant_a": {"percentage": 50, "params": {"max_tokens": 1000}}
            }
        }
    ]
    ```

=== "📊 Step 8: Monitoring & Observability"

    ## Key Metrics to Track

    ### 1. Latency Metrics

    ```
    Time to First Token (TTFT):
    - p50: < 300ms
    - p95: < 500ms
    - p99: < 1000ms

    Token Throughput (tokens/sec per request):
    - p50: > 40 tokens/sec
    - p95: > 30 tokens/sec

    End-to-End Latency:
    - Short responses (< 100 tokens): < 5s
    - Medium responses (< 500 tokens): < 20s
    - Long responses (1000+ tokens): < 40s
    ```

    ### 2. GPU Metrics

    ```
    GPU Utilization:
    - Target: 70-85% average
    - Alert if < 50% (underutilized) or > 95% (overloaded)

    Batch Size:
    - Average batch size: 32-64
    - Alert if < 8 (inefficient batching)

    KV Cache Hit Rate:
    - Target: > 40%
    - Tracks prefix caching effectiveness

    Queue Depth:
    - Target: < 100 requests
    - Alert if > 500 (need to scale)
    ```

    ### 3. Business Metrics

    ```
    Request Success Rate:
    - Target: > 99.5%
    - Track by error type (rate limit, timeout, model error)

    Token Usage:
    - Daily active tokens processed
    - Cost per token
    - Cache savings (% tokens served from cache)

    User Engagement:
    - Messages per conversation
    - Active conversations per user
    - Retry rate (user regenerates response)
    ```

    ### 4. Monitoring Implementation

    ```python
    from prometheus_client import Counter, Histogram, Gauge
    import time

    # Define metrics
    request_counter = Counter(
        'chat_requests_total',
        'Total chat requests',
        ['model', 'status']
    )

    latency_histogram = Histogram(
        'chat_latency_seconds',
        'Request latency',
        ['model', 'endpoint'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )

    token_counter = Counter(
        'tokens_processed_total',
        'Total tokens processed',
        ['model', 'type']  # type: prompt or completion
    )

    gpu_utilization = Gauge(
        'gpu_utilization_percent',
        'GPU utilization',
        ['gpu_id', 'node']
    )

    queue_depth = Gauge(
        'request_queue_depth',
        'Pending requests in queue',
        ['model']
    )

    # Instrumentation
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        start_time = time.time()

        try:
            # Process request
            response = await generate_response(request)

            # Record metrics
            request_counter.labels(
                model=request.model,
                status='success'
            ).inc()

            token_counter.labels(
                model=request.model,
                type='prompt'
            ).inc(response.usage.prompt_tokens)

            token_counter.labels(
                model=request.model,
                type='completion'
            ).inc(response.usage.completion_tokens)

            return response

        except Exception as e:
            request_counter.labels(
                model=request.model,
                status='error'
            ).inc()
            raise

        finally:
            latency = time.time() - start_time
            latency_histogram.labels(
                model=request.model,
                endpoint='chat_completions'
            ).observe(latency)
    ```

    ### 5. Alerting Rules

    ```yaml
    # Prometheus alerting rules
    groups:
      - name: chatgpt_alerts
        rules:
          # High error rate
          - alert: HighErrorRate
            expr: |
              rate(chat_requests_total{status="error"}[5m])
              / rate(chat_requests_total[5m]) > 0.01
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"

          # High latency
          - alert: HighLatency
            expr: |
              histogram_quantile(0.95,
                rate(chat_latency_seconds_bucket[5m])
              ) > 5.0
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "p95 latency above 5 seconds"

          # GPU underutilization
          - alert: LowGPUUtilization
            expr: avg(gpu_utilization_percent) < 50
            for: 15m
            labels:
              severity: warning
            annotations:
              summary: "GPU utilization below 50%"

          # Request queue building up
          - alert: HighQueueDepth
            expr: request_queue_depth > 500
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Request queue depth above 500"

          # Rate limit hit rate high
          - alert: HighRateLimitRate
            expr: |
              rate(chat_requests_total{status="rate_limited"}[5m])
              / rate(chat_requests_total[5m]) > 0.1
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "More than 10% of requests rate limited"
    ```

    ### 6. Distributed Tracing

    ```python
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer(__name__)

    async def process_chat_request(request: ChatRequest):
        """Process chat request with distributed tracing"""

        with tracer.start_as_current_span("chat_request") as span:
            span.set_attribute("user_id", request.user_id)
            span.set_attribute("model", request.model)
            span.set_attribute("message_count", len(request.messages))

            # Rate limiting
            with tracer.start_as_current_span("rate_limit_check"):
                await check_rate_limit(request.user_id)

            # Content moderation
            with tracer.start_as_current_span("content_moderation"):
                await moderate_content(request.messages)

            # LLM inference
            with tracer.start_as_current_span("llm_inference") as llm_span:
                llm_span.set_attribute("input_tokens", count_tokens(request.messages))
                response = await generate_response(request)
                llm_span.set_attribute("output_tokens", response.usage.completion_tokens)

            # Usage tracking
            with tracer.start_as_current_span("usage_tracking"):
                await track_usage(request, response)

            span.set_status(Status(StatusCode.OK))
            return response
    ```

=== "🎤 Step 9: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5-10 min)

    **Key Questions to Ask:**

    ```
    Scale & Traffic:
    - How many daily active users?
    - What's the peak QPS?
    - Average conversation length (messages)?
    - Average message size (tokens)?

    Functional Requirements:
    - Multi-turn conversations required?
    - Streaming responses or batch?
    - Need conversation history persistence?
    - Tool/function calling support?
    - Multi-modal (images, files)?

    Non-Functional:
    - Latency requirements (TTFT, total)?
    - Availability SLA?
    - Consistency requirements?
    - Budget constraints (GPU cost)?

    Constraints:
    - Specific model family (GPT-4, Claude, Llama)?
    - Model size (billions of parameters)?
    - Context window (8K, 32K, 128K tokens)?
    - Rate limiting strategy?
    ```

    ### 2. Present High-Level Design (10-15 min)

    **Start with simplest version:**

    ```
    1. Client → API Gateway → LLM Service → Response
    2. Add conversation storage
    3. Add streaming support
    4. Add rate limiting
    5. Add caching
    ```

    **Draw components:**
    - API layer (load balancer, auth, rate limiting)
    - Inference layer (GPU pools, model serving)
    - Storage layer (conversations, messages, usage)
    - Caching layer (Redis for sessions, rate limits, KV cache)

    ### 3. Deep Dive (25-35 min)

    **Interviewer will likely ask about:**

    1. **LLM Serving:**
       - "How do you maximize GPU utilization?"
       - → Continuous batching with vLLM, 70-85% target utilization

    2. **Streaming:**
       - "How do you implement streaming responses?"
       - → SSE or WebSocket, backpressure handling, chunk-based delivery

    3. **Context Management:**
       - "What if conversation exceeds context window?"
       - → Sliding window (keep recent) or summarization (compress old)

    4. **Rate Limiting:**
       - "How do you prevent abuse?"
       - → Token bucket per user/org, track RPM and TPM

    5. **Scalability:**
       - "How do you handle 10x traffic?"
       - → Horizontal scaling of API, GPU auto-scaling, database sharding

    ### 4. Common Follow-up Questions

    **Q: How do you reduce latency for first token?**
    ```
    A:
    1. Prompt caching (reuse KV cache for common prefixes)
    2. Model optimization (quantization, pruning)
    3. Strategic model placement (closer to users)
    4. Speculative decoding (draft model + verification)
    5. Continuous batching (no wait for batch to fill)
    ```

    **Q: How do you handle model updates without downtime?**
    ```
    A:
    1. Blue-green deployment: Deploy new version alongside old
    2. Gradual rollout: Route 1% → 10% → 50% → 100% to new version
    3. Shadow testing: Run both versions, compare outputs
    4. Quick rollback: Keep old version running for 24h
    ```

    **Q: How do you optimize GPU cost?**
    ```
    A:
    1. Prompt caching (50% compute savings)
    2. Model routing (use cheaper model when possible)
    3. Batching (maximize GPU utilization)
    4. Spot instances (30% cost savings)
    5. Model quantization (int8, int4)
    ```

    **Q: What if GPU node fails during request?**
    ```
    A:
    1. Detect failure: Health check every 10s, mark unhealthy
    2. Retry logic: Client retries with exponential backoff
    3. Request routing: Route new requests to healthy nodes
    4. Graceful degradation: Return partial response if possible
    ```

    **Q: How do you ensure user data privacy?**
    ```
    A:
    1. Encryption: At rest (AES-256) and in transit (TLS 1.3)
    2. Isolation: Strong tenant isolation in database
    3. Audit logging: Track all access to conversations
    4. Data retention: Configurable retention policies
    5. Compliance: GDPR, SOC 2, HIPAA if needed
    ```

    **Q: How do you handle abusive/harmful content?**
    ```
    A:
    1. Input moderation: Filter before sending to LLM
    2. Output moderation: Filter LLM responses
    3. Rate limiting: Prevent spam
    4. User reporting: Allow users to flag content
    5. Pattern detection: ML models to detect abuse
    ```

    ### 5. Dos and Don'ts

    **DO:**
    - ✅ Start simple, add complexity incrementally
    - ✅ Draw diagrams (architecture, data flow)
    - ✅ Use numbers (calculate QPS, storage, cost)
    - ✅ Discuss trade-offs explicitly
    - ✅ Ask clarifying questions
    - ✅ Consider failure scenarios

    **DON'T:**
    - ❌ Jump to complex solution immediately
    - ❌ Ignore scalability from the start
    - ❌ Over-engineer (YAGNI principle)
    - ❌ Forget about costs (GPUs expensive!)
    - ❌ Ignore data privacy/security
    - ❌ Forget to monitor/alert

    ### 6. Red Flags to Avoid

    ```
    ❌ "We'll use the latest/most expensive model for everything"
       → Show cost awareness, model routing

    ❌ "We'll load entire conversation into context every time"
       → Discuss context window management

    ❌ "We'll store everything in one database"
       → Show sharding/partitioning understanding

    ❌ "Streaming is too complex, we'll do batch only"
       → Understand UX implications

    ❌ "We don't need rate limiting initially"
       → Consider abuse prevention from day 1
    ```

=== "📚 Step 10: Real-World Examples"

    ## OpenAI ChatGPT Architecture (Public Info)

    **Scale (as of 2024):**
    - 100M+ weekly active users
    - 2B+ requests per day
    - 500B+ tokens processed per day

    **Key Technologies:**
    - Models: GPT-4, GPT-3.5-turbo, GPT-4-turbo
    - Inference: Custom serving stack (likely vLLM or similar)
    - API: REST API with streaming (SSE)
    - Infrastructure: Azure cloud, NVIDIA A100/H100 GPUs

    **Notable Features:**
    - Streaming responses (SSE)
    - Function calling / plugins
    - Vision support (GPT-4V)
    - Long context (128K tokens for GPT-4-turbo)
    - Prompt caching (automatic)

    **Pricing (2024):**
    ```
    GPT-4:
    - Input: $0.03 / 1K tokens
    - Output: $0.06 / 1K tokens

    GPT-3.5-turbo:
    - Input: $0.0015 / 1K tokens
    - Output: $0.002 / 1K tokens

    Implies significant cost optimization through:
    - Efficient serving (batching, caching)
    - Model distillation
    - Mixed precision inference
    ```

    ---

    ## Anthropic Claude Architecture

    **Key Differences from ChatGPT:**
    - Longer context window (200K tokens for Claude 2.1)
    - Constitutional AI for safety
    - Different prompt format (Human:/Assistant:)
    - Streaming via SSE
    - Similar API design (OpenAI-compatible)

    **Innovations:**
    - Extended context (200K tokens = ~150K words)
    - Better at following instructions
    - Reduced hallucinations (Constitutional AI)

    ---

    ## Technical Insights from Production Systems

    ### 1. Token Caching Impact

    ```
    Without caching:
    - Cost: $0.03/1K input tokens
    - 100M requests/day × 1.5K tokens = 150B tokens/day
    - Cost: $4,500/day = $135K/month

    With 50% cache hit rate:
    - Cached: 75B tokens (free or heavily discounted)
    - Computed: 75B tokens × $0.03/1K = $2,250/day
    - Savings: $2,250/day = $67.5K/month

    Real-world observation:
    - System prompts: 90%+ cache hit
    - Conversation history: 40-60% cache hit
    - Combined: ~50-60% overall cache hit
    ```

    ### 2. Model Selection Impact

    ```
    Scenario: Code explanation

    Option A: Always GPT-4
    - Quality: Excellent
    - Cost: $0.09 per request (avg 2K tokens)
    - Latency: 3-5 seconds

    Option B: Route to GPT-3.5 for simple code, GPT-4 for complex
    - Quality: Good for 80% of requests, excellent for 20%
    - Cost: 80% × $0.006 + 20% × $0.09 = $0.023 per request
    - Savings: 74% cost reduction

    Implementation:
    - Classify query complexity
    - Route simple queries to cheaper model
    - Use expensive model only when needed
    ```

    ### 3. Batching Efficiency

    ```
    Static batching (batch size 32):
    - Wait for 32 requests to arrive
    - Process batch
    - Worst case: First request waits for 31 others
    - Average wait: 500ms+
    - GPU utilization: 60% (due to padding)

    Continuous batching (vLLM):
    - Start processing immediately
    - Add requests to batch dynamically
    - Remove completed requests
    - Average wait: < 50ms
    - GPU utilization: 85%

    Impact:
    - 40% higher throughput
    - 90% lower latency for small requests
    - Same GPU hardware
    ```

    ### 4. Rate Limiting Learnings

    ```
    Initial approach: Simple counter
    - Problem: Bursty traffic exhausts quota instantly
    - User experience: Unpredictable failures

    Token bucket:
    - Refill at steady rate
    - Allow bursts up to capacity
    - Smooth degradation

    Real-world quotas (OpenAI-like):
    - Free tier: 3 RPM, 40K TPM
    - Pro ($20/month): 60 RPM, 90K TPM
    - Enterprise: Custom (10K+ RPM)

    Observation:
    - 80% of users stay within free tier
    - 15% upgrade to pro for higher limits
    - 5% need enterprise
    ```

    ### 5. Streaming Benefits

    ```
    Batch response:
    - User sees: Loading spinner for 15 seconds
    - Then: Full response appears
    - Perceived latency: 15 seconds

    Streaming response:
    - User sees: First word after 500ms
    - Then: Continuous text stream
    - Perceived latency: 500ms

    User satisfaction:
    - Batch: 70% satisfied
    - Streaming: 92% satisfied
    - Reason: Immediate feedback, can read while generating
    ```

---

## Summary

**Key Takeaways:**

1. **LLM serving is the core challenge**: vLLM with continuous batching and prompt caching is critical for efficiency
2. **Streaming is UX-critical**: Users expect immediate feedback, implement SSE or WebSocket
3. **Context management matters**: Sliding window or summarization for long conversations
4. **Rate limiting is essential**: Token bucket algorithm per user/org
5. **Cost optimization is key**: Prompt caching, model routing, efficient batching
6. **Multi-tenancy requires care**: Strong isolation, no data leakage
7. **Monitoring is critical**: Track TTFT, GPU utilization, error rates

**Estimated Costs at Scale:**

```
100M DAU, 2B requests/day, 4T tokens/day:

GPU costs: $4.3M/month (2,400 A100 GPUs)
Storage: $50K/month (1.6 PB)
Bandwidth: $20K/month
Other infrastructure: $100K/month

Total: ~$4.5M/month = $54M/year

With optimizations (caching, routing, spots):
- GPU: -40% = $2.6M/month
- Total: ~$2.8M/month = $33.6M/year

Revenue target (to be profitable):
- Need ~$5M/month revenue
- At $20/month/user: 250K paying users (0.25% conversion)
- Or API usage pricing
```

This is a hard problem that requires deep understanding of:
- Large-scale distributed systems
- GPU optimization and serving
- Real-time streaming protocols
- Database sharding and scaling
- Cost optimization strategies
- Security and privacy
