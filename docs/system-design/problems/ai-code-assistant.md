# Design AI Code Assistant (GitHub Copilot / Cursor)

An AI-powered code completion and assistance system that provides real-time code suggestions, multi-line completions, chat-based code generation, and context-aware recommendations directly within IDEs.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M developers, 10B completions/day, 500M chat messages/day, <300ms p95 latency |
| **Key Challenges** | Low-latency inference, context extraction, multi-file understanding, caching strategies, GPU optimization |
| **Core Concepts** | LSP integration, prompt prefix caching, speculative decoding, KV cache, Fill-in-the-Middle (FIM), token streaming |
| **Companies** | GitHub (Copilot), Cursor, Replit (Ghostwriter), Amazon (CodeWhisperer), Tabnine, Codeium |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Inline Code Completion** | Autocomplete as user types, single/multi-line suggestions | P0 (Must have) |
    | **Fill-in-the-Middle** | Complete code within functions/blocks (not just suffix) | P0 (Must have) |
    | **Chat-based Code Generation** | Natural language to code with multi-turn conversations | P0 (Must have) |
    | **Multi-file Context** | Understand imports, related files, project structure | P0 (Must have) |
    | **Real-time Context Extraction** | Track cursor position, recent edits, open files | P0 (Must have) |
    | **Multiple Language Support** | Python, JavaScript, TypeScript, Java, Go, Rust, etc. | P0 (Must have) |
    | **IDE Integration** | VSCode, JetBrains, Vim, Visual Studio plugins | P0 (Must have) |
    | **Code Explanation** | Explain selected code snippets | P1 (Should have) |
    | **Bug Detection** | Identify potential errors, security issues | P1 (Should have) |
    | **Test Generation** | Generate unit tests for functions | P1 (Should have) |
    | **Refactoring Suggestions** | Improve code quality, suggest better patterns | P1 (Should have) |
    | **Credential Scanning** | Block suggestions containing API keys, secrets | P1 (Should have) |
    | **Telemetry & Analytics** | Track acceptance rate, latency, user satisfaction | P1 (Should have) |
    | **Codebase Indexing** | Vector embeddings for semantic code search | P2 (Nice to have) |
    | **Collaborative Filtering** | Learn from team patterns, project-specific completions | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (assume pre-trained code models)
    - IDE development itself (focus on plugin architecture)
    - Full static analysis engine (use existing linters)
    - Code execution environment (use sandboxed runners if needed)
    - Git operations and version control
    - Code review platform integration

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Inline Completion)** | < 300ms p95 | Users expect instant suggestions while typing |
    | **Latency (Chat Response TTFT)** | < 500ms p95 | Chat can tolerate slightly higher latency |
    | **Token Throughput** | > 50 tokens/sec | Smooth streaming for multi-line completions |
    | **Availability** | 99.95% uptime | Developers depend on it for productivity |
    | **Context Window** | 8K-32K tokens | Sufficient for most file contexts + imports |
    | **Acceptance Rate** | > 30% for completions | Measure of suggestion quality |
    | **GPU Utilization** | > 80% average | Expensive GPUs must be maximally utilized |
    | **Cache Hit Rate** | > 70% for context | Reuse prompt prefixes to reduce latency |
    | **Security** | 100% credential detection | Never suggest code with leaked secrets |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Users:
    - Daily Active Users (DAU): 50M developers
    - Monthly Active Users (MAU): 80M developers
    - Peak concurrent users: 10M (20% of DAU)

    Code completions (inline suggestions):
    - Completions per developer per day: 200 completions
    - Daily completions: 50M × 200 = 10B completions/day
    - Completion QPS: 10B / 86,400 = ~115,740 req/sec average
    - Peak QPS: 2x average = ~230,000 req/sec
    - Average latency budget: 200ms
    - 80% of completions are single-line, 20% multi-line

    Chat messages:
    - Developers using chat: 40% of DAU = 20M
    - Messages per developer per day: 25 messages
    - Daily chat messages: 20M × 25 = 500M messages/day
    - Chat QPS: 500M / 86,400 = ~5,787 req/sec
    - Peak chat QPS: 3x average = ~17,361 req/sec

    Token processing:
    - Average input tokens per completion: 2,000 tokens (file context + imports)
    - Average output tokens per completion: 100 tokens (suggestion)
    - Total tokens per completion: 2,100 tokens
    - Daily completion tokens: 10B × 2,100 = 21T tokens/day
    - Token throughput: 21T / 86,400 = ~243M tokens/sec

    Chat token processing:
    - Average input tokens per chat: 4,000 tokens (conversation + code context)
    - Average output tokens per chat: 500 tokens (explanation/generation)
    - Daily chat tokens: 500M × 4,500 = 2.25T tokens/day
    - Chat token throughput: 2.25T / 86,400 = ~26M tokens/sec

    Total token throughput: 243M + 26M = ~269M tokens/sec average

    Cache efficiency:
    - 70% of completion contexts are cache hits (repeated file prefixes)
    - Effective new tokens: 243M × 0.3 = ~73M tokens/sec
    ```

    ### Storage Estimates

    ```
    User data:
    - 80M MAU × 5 KB (profile, preferences, API keys) = 400 GB

    Session data (active contexts):
    - 10M concurrent users × 50 KB (current file, cursor, recent edits) = 500 GB
    - In-memory cache: 500 GB distributed across cache tier

    Completion logs (for analytics):
    - Completions per day: 10B
    - Log entry size: 500 bytes (prompt hash, completion, accepted, latency)
    - Daily logs: 10B × 500 bytes = 5 TB/day
    - 30-day retention: 5 TB × 30 = 150 TB
    - With compression (3x): 50 TB

    Chat conversation history:
    - Conversations per day: 500M messages
    - Average message size: 3 KB
    - Daily storage: 500M × 3 KB = 1.5 TB/day
    - 90-day retention: 1.5 TB × 90 = 135 TB

    Telemetry data (acceptance rate, latency):
    - Events per day: 10B completions + 500M chats = 10.5B events
    - Event size: 200 bytes
    - Daily: 10.5B × 200 bytes = 2.1 TB/day
    - 1-year retention: 2.1 TB × 365 = 766 TB
    - Aggregated/sampled: 76 TB

    Prompt cache (KV cache persistence):
    - Active file contexts: 50M hot contexts
    - KV cache per context: 20 KB (compressed)
    - Total: 50M × 20 KB = 1 TB (in Redis/SSD)

    Model weights:
    - Small completion model (1B params): 2 GB per copy
    - Medium chat model (7B params): 14 GB per copy
    - Large chat model (34B params): 68 GB per copy
    - 500 replicas × 2 GB (completion) = 1 TB
    - 200 replicas × 14 GB (chat) = 2.8 TB
    - Total: ~4 TB for model serving

    Code embeddings (vector index):
    - Indexed code snippets: 10B snippets
    - Embedding size: 768 dimensions × 4 bytes = 3 KB
    - Total: 10B × 3 KB = 30 TB
    - Vector database (quantized): 10 TB

    Total storage: 400 GB (users) + 50 TB (completion logs) + 135 TB (chat history) +
                   76 TB (telemetry) + 1 TB (cache) + 4 TB (models) + 10 TB (embeddings)
                   ≈ 276 TB
    ```

    ### Compute Estimates (GPU)

    ```
    Completion model (small, 1B parameters, optimized for speed):
    - Request QPS: 115,740 req/sec
    - Cache hit rate: 70% (prompt prefix cached)
    - Effective new token processing: 115,740 × 2,000 × 0.3 = ~69M tokens/sec
    - Throughput per A100 GPU (with speculative decoding): ~50,000 tokens/sec
    - GPUs needed: 69M / 50,000 = 1,380 A100 GPUs
    - With batching efficiency (5x): ~280 A100 GPUs

    Chat model (medium, 7B parameters):
    - Request QPS: 5,787 req/sec
    - Tokens per second: 5,787 × 4,500 = 26M tokens/sec
    - Throughput per A100 GPU: ~10,000 tokens/sec
    - GPUs needed: 26M / 10,000 = 2,600 A100 GPUs
    - With batching (3x): ~870 A100 GPUs

    Total GPUs: 280 + 870 = 1,150 A100 GPUs (80GB)
    Cost: 1,150 × $2.50/hour = $2,875/hour = $69,000/day = $2.07M/month

    Peak capacity (2-3x):
    - Peak GPUs: 1,150 × 2.5 = 2,875 A100 GPUs
    - Auto-scaling with spot instances to reduce cost

    GPU memory per instance:
    - Model weights: 2 GB (completion) or 14 GB (chat)
    - KV cache per request: 10 MB (completion), 30 MB (chat)
    - Batch size: 64-128 concurrent requests
    - Memory per GPU: 14 GB + (128 × 30 MB) = ~18 GB (fits in A100 80GB)
    ```

    ### Bandwidth Estimates

    ```
    Request ingress:
    - Completions: 115,740 req/sec × 8 KB (context) = 926 MB/sec ≈ 7.4 Gbps
    - Chat: 5,787 req/sec × 16 KB (context) = 93 MB/sec ≈ 740 Mbps
    - Total ingress: ~8.1 Gbps

    Response egress:
    - Completions: 115,740 req/sec × 400 bytes (suggestion) = 46 MB/sec ≈ 370 Mbps
    - Chat (streaming): 5,787 req/sec × 2 KB = 11.5 MB/sec ≈ 92 Mbps
    - Total egress: ~462 Mbps

    IDE plugin heartbeat:
    - Active connections: 10M
    - Heartbeat every 30s: 10M / 30 = 333K connections/sec
    - Heartbeat size: 100 bytes
    - Bandwidth: 333K × 100 bytes = 33 MB/sec ≈ 264 Mbps

    Internal (cache layer ↔ GPU):
    - Cache requests: 243M tokens/sec × 8 bytes = 1.9 GB/sec ≈ 15.2 Gbps
    - Cache responses: 46 MB/sec

    Total ingress: ~8.1 Gbps
    Total egress: ~726 Mbps
    Internal: ~15.2 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Prompt prefix cache (GPU KV cache):
    - Hot contexts: 5M concurrent active files
    - KV cache per context: 50 MB
    - Total: 5M × 50 MB = 250 TB (distributed across GPU memory)

    Redis completion cache:
    - Recent completions: 100M cached completions
    - Average size: 2 KB (prompt fingerprint → suggestion)
    - Total: 100M × 2 KB = 200 GB
    - TTL: 1 hour

    Context extraction cache:
    - Parsed AST for open files: 10M files
    - AST size: 100 KB per file
    - Total: 10M × 100 KB = 1 TB
    - In-memory distributed cache

    User session cache:
    - Active sessions: 10M users
    - Session data: 50 KB (auth, preferences, current context)
    - Total: 10M × 50 KB = 500 GB

    Total cache memory: 200 GB (completions) + 1 TB (AST) + 500 GB (sessions)
                        = 1.7 TB
    ```

    ---

    ## Key Assumptions

    1. Most completions are single-line (80%), multi-line less frequent (20%)
    2. Cache hit rate of 70% for prompt prefixes (repeated file contexts)
    3. Average acceptance rate of 30-40% for suggestions
    4. Developers use completion ~200 times per day, chat ~25 times per day
    5. Context window limited to 8K-32K tokens (current file + imports)
    6. Low latency (<300ms) is critical for inline completion UX
    7. GPU inference optimized with speculative decoding, prompt caching
    8. Most traffic during business hours (9am-6pm across timezones)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Low-latency inference:** Aggressive caching, speculative decoding, optimized models (<1B params for completion)
    2. **Context-aware:** Extract relevant code context (imports, definitions, recent edits)
    3. **Streaming responses:** Token-by-token streaming for smooth UX
    4. **Multi-tier caching:** Completion cache, prompt prefix cache, KV cache
    5. **Credential scanning:** Real-time detection of API keys, passwords in suggestions
    6. **Telemetry-driven:** Log acceptance rate, latency to optimize model quality

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "IDE Layer"
            VSCode[VSCode Extension<br/>LSP Client<br/>Context Extraction]
            JetBrains[JetBrains Plugin<br/>IntelliJ/PyCharm]
            Vim[Vim/Neovim Plugin]
        end

        subgraph "API Gateway & Load Balancer"
            Gateway[API Gateway<br/>Auth, Rate Limiting<br/>WebSocket/SSE]
            LB[Load Balancer<br/>Request Routing<br/>Health Checks]
        end

        subgraph "Completion Service"
            CompletionAPI[Completion API<br/>FIM Formatting<br/>Prompt Construction]
            ChatAPI[Chat API<br/>Conversation Management<br/>Multi-turn Context]
            ContextService[Context Extraction Service<br/>AST Parsing<br/>Import Resolution<br/>Multi-file Context]
        end

        subgraph "Caching Layer"
            CompletionCache[(Completion Cache<br/>Redis<br/>TTL: 1 hour)]
            PrefixCache[(Prompt Prefix Cache<br/>Redis/SSD<br/>KV Cache Persistence)]
            SessionCache[(Session Cache<br/>User Context<br/>Active Files)]
        end

        subgraph "LLM Inference Cluster"
            InferenceRouter[Inference Router<br/>Model Selection<br/>GPU Load Balancing]

            subgraph "Completion Model Pool"
                CompModel1[Small Model<br/>1B params<br/>vLLM<br/>Speculative Decoding]
                CompModel2[Small Model<br/>1B params<br/>vLLM]
                CompModelN[Small Model<br/>...]
            end

            subgraph "Chat Model Pool"
                ChatModel1[Medium Model<br/>7B params<br/>vLLM]
                ChatModel2[Medium Model<br/>7B params]
                ChatModelN[Medium Model<br/>...]
            end
        end

        subgraph "Post-Processing"
            SecurityFilter[Security Filter<br/>Credential Scanner<br/>Regex + ML Detection]
            RankingService[Ranking Service<br/>Score Suggestions<br/>Multi-candidate Selection]
        end

        subgraph "Analytics & Monitoring"
            TelemetryCollector[Telemetry Collector<br/>Kafka<br/>Acceptance Rate<br/>Latency Metrics]
            MetricsDB[(Metrics Database<br/>ClickHouse<br/>Aggregated Analytics)]
            Monitoring[Monitoring<br/>Prometheus/Grafana<br/>GPU Utilization<br/>Cache Hit Rate]
        end

        subgraph "Storage"
            UserDB[(User Database<br/>PostgreSQL<br/>Profiles, API Keys)]
            LogStore[(Log Storage<br/>S3/GCS<br/>Completion Logs)]
            VectorDB[(Vector Database<br/>Pinecone/Weaviate<br/>Code Embeddings)]
        end

        VSCode -->|HTTPS/WebSocket| Gateway
        JetBrains -->|HTTPS/WebSocket| Gateway
        Vim -->|HTTPS/WebSocket| Gateway

        Gateway --> LB
        LB --> CompletionAPI
        LB --> ChatAPI

        CompletionAPI --> ContextService
        ChatAPI --> ContextService

        CompletionAPI --> CompletionCache
        CompletionAPI --> PrefixCache
        ChatAPI --> SessionCache

        CompletionCache -->|Miss| InferenceRouter
        CompletionAPI --> InferenceRouter

        InferenceRouter --> CompModel1
        InferenceRouter --> CompModel2
        InferenceRouter --> CompModelN
        InferenceRouter --> ChatModel1
        InferenceRouter --> ChatModel2
        InferenceRouter --> ChatModelN

        CompModel1 --> SecurityFilter
        ChatModel1 --> SecurityFilter

        SecurityFilter --> RankingService
        RankingService --> CompletionAPI

        CompletionAPI --> TelemetryCollector
        ChatAPI --> TelemetryCollector

        TelemetryCollector --> MetricsDB
        Monitoring --> MetricsDB
        Monitoring --> InferenceRouter

        CompletionAPI --> UserDB
        TelemetryCollector --> LogStore
        ContextService --> VectorDB

        style VSCode fill:#e1f5ff
        style CompletionAPI fill:#ffe1e1
        style ChatAPI fill:#ffe1e1
        style InferenceRouter fill:#fff4e1
        style CompModel1 fill:#e8f5e9
        style ChatModel1 fill:#e8f5e9
        style CompletionCache fill:#f3e5f5
        style PrefixCache fill:#f3e5f5
        style SecurityFilter fill:#ffebee
        style TelemetryCollector fill:#e0f2f1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **LSP Integration** | Standard protocol for IDE communication, AST parsing, symbol resolution | Custom protocol per IDE (too much work), direct file parsing (less accurate) |
    | **Fill-in-the-Middle (FIM)** | Allow completions within code blocks, not just suffix | Prefix-only completion (limited to end-of-line), infilling models (slower) |
    | **Prompt Prefix Caching** | Reuse KV cache for repeated file contexts (70% hit rate) | No caching (4x higher latency), full completion caching (lower hit rate) |
    | **Speculative Decoding** | Speed up inference by 2-3x for common tokens | Standard autoregressive decoding (slower), distillation (quality loss) |
    | **Small Completion Model** | <100ms inference latency, fits 64+ batch size per GPU | Large model (too slow), no model (just templates, poor quality) |
    | **Credential Scanner** | Prevent leaking API keys, passwords in suggestions | No scanning (security risk), client-side only (bypassable) |
    | **Multi-tier Caching** | Completion cache (exact), prefix cache (partial), KV cache (token-level) | Single cache tier (lower hit rate), no caching (too slow) |
    | **Streaming Responses** | Show tokens as generated, better UX for multi-line | Wait for full completion (slow, bad UX), batch responses (not real-time) |

    **Key Trade-off:** We chose **latency over quality** for inline completions. A smaller, faster model (1B params) provides instant suggestions, while chat uses a larger model (7B params) where latency is less critical.

    ---

    ## Data Flow

    ### Inline Code Completion Flow

    ```
    1. User types in IDE → Trigger completion event
    2. IDE Extension extracts context:
       - Current file content (up to 8K tokens)
       - Cursor position
       - Recent edits (last 5 minutes)
       - Imported modules/files
       - Function signatures in scope
    3. Send request to Completion API:
       POST /v1/completions
       {
         "context": "...",
         "cursor": 42,
         "language": "python",
         "recent_edits": [...]
       }
    4. Completion API checks Completion Cache (Redis):
       - Hash: SHA256(context + cursor + language)
       - If hit (30% chance): Return cached suggestion (5ms)
    5. If miss, check Prompt Prefix Cache:
       - Hash: SHA256(file_path + first_N_lines)
       - If hit (70% chance): Reuse KV cache, only process new tokens
    6. Construct FIM prompt:
       <prefix>current_file_before_cursor</prefix>
       <suffix>current_file_after_cursor</suffix>
       <middle>
    7. Send to Inference Router → Select GPU with lowest queue
    8. Model inference (vLLM with speculative decoding):
       - Batch with other requests (batch size: 64)
       - Generate 50-100 tokens (multi-line completion)
       - Time: 100-200ms
    9. Post-process:
       - Credential Scanner: Regex + ML model check for secrets
       - If detected: Block suggestion, log incident
       - Ranking: Score multiple candidates, pick best
    10. Return suggestion to IDE (total: <300ms)
    11. User accepts/rejects → Log to Telemetry
    ```

    ### Chat-based Code Generation Flow

    ```
    1. User opens chat panel, types message: "Write a binary search function"
    2. IDE Extension gathers context:
       - Current file
       - Open files in workspace
       - Project language/framework
       - Previous chat history (last 5 messages)
    3. Send request to Chat API:
       POST /v1/chat/completions
       {
         "message": "Write a binary search function",
         "context": {...},
         "conversation_id": "uuid",
         "stream": true
       }
    4. Chat API loads conversation history from Session Cache
    5. Construct multi-turn prompt:
       <system>You are an expert programmer...</system>
       <conversation_history>...</conversation_history>
       <code_context>...</code_context>
       <user>Write a binary search function</user>
    6. Send to Inference Router → Chat Model Pool
    7. Model generates response (streaming):
       - Token-by-token generation
       - Send via SSE/WebSocket
       - User sees response in real-time
    8. Security scan on full response
    9. User accepts → Insert code into file
    10. Log telemetry: response time, tokens, accepted
    ```

=== "🔌 Step 3: API Design"

    ## 1. Code Completion API

    **Endpoint:** `POST /v1/completions`

    **Request:**
    ```json
    {
      "context": {
        "file_path": "/workspace/src/utils.py",
        "language": "python",
        "content": "def calculate_sum(a, b):\n    # Calculate sum",
        "cursor_position": 42,
        "cursor_line": 2,
        "cursor_column": 18
      },
      "recent_edits": [
        {
          "timestamp": "2026-02-05T10:30:00Z",
          "range": {"start": 40, "end": 42},
          "text": "# "
        }
      ],
      "imports": [
        "import numpy as np",
        "from typing import List"
      ],
      "max_tokens": 100,
      "temperature": 0.2,
      "stream": false,
      "options": {
        "multi_line": true,
        "include_imports": false
      }
    }
    ```

    **Response (Non-streaming):**
    ```json
    {
      "id": "compl_abc123xyz",
      "created_at": "2026-02-05T10:30:01Z",
      "model": "code-completion-1b-v2",
      "suggestions": [
        {
          "text": "return a + b",
          "score": 0.95,
          "tokens": 5,
          "language": "python"
        },
        {
          "text": "result = a + b\n    return result",
          "score": 0.78,
          "tokens": 9,
          "language": "python"
        }
      ],
      "latency_ms": 185,
      "cached": false,
      "cache_hit_type": "prefix"
    }
    ```

    **Response (Streaming):**
    ```
    event: completion_start
    data: {"id": "compl_abc123xyz", "model": "code-completion-1b-v2"}

    event: token
    data: {"text": "return", "index": 0}

    event: token
    data: {"text": " a", "index": 1}

    event: token
    data: {"text": " +", "index": 2}

    event: token
    data: {"text": " b", "index": 3}

    event: completion_end
    data: {"tokens": 5, "latency_ms": 195}
    ```

    **Error Codes:**
    - `400`: Invalid context or malformed request
    - `401`: Unauthorized (invalid API key)
    - `429`: Rate limit exceeded
    - `500`: Inference service error
    - `503`: Service temporarily unavailable

    ---

    ## 2. Chat Completion API

    **Endpoint:** `POST /v1/chat/completions`

    **Request:**
    ```json
    {
      "conversation_id": "conv_xyz789",
      "message": "Add error handling to this function",
      "context": {
        "file_path": "/workspace/src/api.py",
        "language": "python",
        "selected_code": "def fetch_data(url):\n    response = requests.get(url)\n    return response.json()",
        "workspace_files": [
          "/workspace/src/api.py",
          "/workspace/tests/test_api.py"
        ]
      },
      "conversation_history": [
        {
          "role": "user",
          "content": "How do I handle exceptions?",
          "timestamp": "2026-02-05T10:25:00Z"
        },
        {
          "role": "assistant",
          "content": "You can use try-except blocks...",
          "timestamp": "2026-02-05T10:25:05Z"
        }
      ],
      "model": "code-chat-7b-v2",
      "max_tokens": 1000,
      "temperature": 0.4,
      "stream": true
    }
    ```

    **Response (Streaming SSE):**
    ```
    event: message_start
    data: {"id": "msg_def456", "conversation_id": "conv_xyz789"}

    event: content_block_start
    data: {"index": 0, "type": "text"}

    event: content_block_delta
    data: {"index": 0, "delta": {"text": "Here's"}}

    event: content_block_delta
    data: {"index": 0, "delta": {"text": " the function with error handling:\n\n"}}

    event: content_block_delta
    data: {"index": 0, "delta": {"text": "```python\n"}}

    event: content_block_delta
    data: {"index": 0, "delta": {"text": "def fetch_data(url):\n    try:\n        response = requests.get(url, timeout=10)\n        response.raise_for_status()\n        return response.json()\n    except requests.exceptions.RequestException as e:\n        logger.error(f\"Failed to fetch data: {e}\")\n        return None\n```"}}

    event: content_block_end
    data: {"index": 0}

    event: message_end
    data: {"tokens": 87, "latency_ms": 3420}
    ```

    ---

    ## 3. Context Extraction API (Internal)

    **Endpoint:** `POST /internal/v1/context/extract`

    **Request:**
    ```json
    {
      "file_path": "/workspace/src/main.py",
      "language": "python",
      "content": "import os\nimport sys\n\ndef main():\n    print('Hello')\n",
      "cursor_position": 45,
      "include_imports": true,
      "include_definitions": true,
      "max_context_tokens": 8000
    }
    ```

    **Response:**
    ```json
    {
      "context": {
        "imports": [
          {"module": "os", "line": 1},
          {"module": "sys", "line": 2}
        ],
        "definitions": [
          {
            "name": "main",
            "type": "function",
            "line": 4,
            "signature": "def main():"
          }
        ],
        "related_files": [
          {
            "path": "/workspace/src/utils.py",
            "reason": "imported_by_current_file",
            "relevance_score": 0.9
          }
        ],
        "symbols_in_scope": ["os", "sys", "main"],
        "recent_edits": [
          {
            "line": 5,
            "content": "    print('Hello')",
            "timestamp": "2026-02-05T10:30:00Z"
          }
        ],
        "total_tokens": 1520
      }
    }
    ```

    ---

    ## 4. Telemetry API (Internal)

    **Endpoint:** `POST /internal/v1/telemetry/events`

    **Request:**
    ```json
    {
      "events": [
        {
          "event_type": "completion_shown",
          "completion_id": "compl_abc123xyz",
          "user_id": "user_12345",
          "timestamp": "2026-02-05T10:30:01.500Z",
          "language": "python",
          "model": "code-completion-1b-v2",
          "latency_ms": 185,
          "cached": false,
          "suggestion_length": 12,
          "context_tokens": 2000
        },
        {
          "event_type": "completion_accepted",
          "completion_id": "compl_abc123xyz",
          "user_id": "user_12345",
          "timestamp": "2026-02-05T10:30:02.200Z",
          "accepted_length": 12,
          "time_to_accept_ms": 700
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "accepted": 2,
      "rejected": 0
    }
    ```

=== "💾 Step 4: Database Design"

    ## Schema Design

    ### 1. Users Table (PostgreSQL)

    ```sql
    CREATE TABLE users (
        user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        api_key_hash VARCHAR(128) UNIQUE NOT NULL,
        organization_id UUID REFERENCES organizations(organization_id),
        tier VARCHAR(50) DEFAULT 'free', -- free, pro, team, enterprise
        created_at TIMESTAMP DEFAULT NOW(),
        last_active_at TIMESTAMP,
        preferences JSONB DEFAULT '{}', -- IDE settings, model preferences

        INDEX idx_users_email (email),
        INDEX idx_users_org (organization_id),
        INDEX idx_users_last_active (last_active_at)
    );
    ```

    ### 2. Completions Log Table (ClickHouse - Analytics)

    ```sql
    CREATE TABLE completions_log (
        completion_id String,
        user_id String,
        organization_id String,
        timestamp DateTime64(3),

        -- Context
        language String,
        file_path String,
        context_tokens UInt32,

        -- Model
        model_name String,
        model_version String,

        -- Performance
        latency_ms UInt32,
        cached Boolean,
        cache_hit_type Enum8('none' = 0, 'full' = 1, 'prefix' = 2),

        -- Suggestion
        suggestion_text String,
        suggestion_tokens UInt32,
        suggestion_score Float32,

        -- User interaction
        shown Boolean DEFAULT true,
        accepted Boolean DEFAULT false,
        time_to_accept_ms Nullable(UInt32),

        -- Security
        security_scan_passed Boolean,
        security_issues String DEFAULT '',

        date Date MATERIALIZED toDate(timestamp)
    )
    ENGINE = MergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (date, user_id, timestamp)
    TTL date + INTERVAL 30 DAY;

    -- Materialized view for acceptance rate
    CREATE MATERIALIZED VIEW completions_acceptance_rate_mv
    ENGINE = AggregatingMergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (date, language, model_name)
    AS SELECT
        date,
        language,
        model_name,
        countState() as total_shown,
        countIfState(accepted = true) as total_accepted,
        avgState(latency_ms) as avg_latency
    FROM completions_log
    GROUP BY date, language, model_name;
    ```

    ### 3. Chat Conversations Table (PostgreSQL)

    ```sql
    CREATE TABLE conversations (
        conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(user_id),
        title VARCHAR(255),
        workspace_path VARCHAR(500),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        message_count INT DEFAULT 0,

        INDEX idx_conversations_user (user_id),
        INDEX idx_conversations_created (created_at DESC)
    );

    CREATE TABLE messages (
        message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL, -- 'user', 'assistant'
        content TEXT NOT NULL,
        code_context JSONB, -- Files, selected code, etc.
        model_name VARCHAR(100),
        tokens_input INT,
        tokens_output INT,
        latency_ms INT,
        created_at TIMESTAMP DEFAULT NOW(),

        INDEX idx_messages_conversation (conversation_id, created_at),
        INDEX idx_messages_created (created_at DESC)
    );
    ```

    ### 4. Session Cache (Redis)

    ```
    Key: session:{user_id}
    Value: {
        "active_files": [
            {
                "path": "/workspace/src/main.py",
                "language": "python",
                "last_cursor_position": 145,
                "ast_cache_key": "ast:abc123"
            }
        ],
        "workspace_path": "/workspace",
        "recent_edits": [...],
        "last_heartbeat": "2026-02-05T10:30:00Z"
    }
    TTL: 1 hour (refreshed on activity)
    ```

    ### 5. Completion Cache (Redis)

    ```
    Key: completion:{hash(context+cursor+language)}
    Value: {
        "suggestions": [
            {"text": "return a + b", "score": 0.95, "tokens": 5}
        ],
        "model": "code-completion-1b-v2",
        "cached_at": "2026-02-05T10:30:01Z"
    }
    TTL: 1 hour

    -- Cache key generation
    hash = SHA256(file_path + content_before_cursor[-2000:] + content_after_cursor[:500] + language)
    ```

    ### 6. Prompt Prefix Cache (Redis + SSD)

    ```
    Key: prefix:{hash(file_path+first_2000_tokens)}
    Value: {
        "kv_cache_path": "s3://cache/kv_12345.bin",
        "token_count": 2000,
        "file_hash": "abc123def456",
        "model_version": "code-completion-1b-v2",
        "created_at": "2026-02-05T10:25:00Z",
        "access_count": 42
    }
    TTL: 24 hours (hot contexts), 7 days (warm contexts)

    -- KV cache stored on SSD/S3, loaded into GPU on cache hit
    ```

    ### 7. Code Embeddings (Vector Database - Pinecone/Weaviate)

    ```json
    {
      "id": "code_snippet_12345",
      "vector": [0.123, -0.456, ...],  // 768 dimensions
      "metadata": {
        "file_path": "/workspace/src/utils.py",
        "function_name": "calculate_sum",
        "language": "python",
        "code": "def calculate_sum(a, b):\n    return a + b",
        "docstring": "Calculate sum of two numbers",
        "imports": ["numpy", "typing"],
        "last_updated": "2026-02-05T10:00:00Z"
      }
    }
    ```

    ### 8. Rate Limiting (Redis)

    ```
    -- Token bucket algorithm
    Key: rate_limit:{user_id}:completions
    Value: {
        "tokens": 950,
        "last_refill": "2026-02-05T10:30:00Z",
        "bucket_size": 1000,
        "refill_rate": 100  // per minute
    }
    TTL: 1 hour

    Key: rate_limit:{user_id}:chat
    Value: {
        "tokens": 45,
        "bucket_size": 50,
        "refill_rate": 10  // per minute
    }
    TTL: 1 hour
    ```

=== "🔍 Step 5: Deep Dive - Key Components"

    ## 1. Context Extraction & LSP Integration

    **Challenge:** Extract relevant code context from IDE without sending entire project.

    **Solution:** Integrate with Language Server Protocol (LSP) for precise context.

    ### LSP Integration Code (VSCode Extension)

    ```typescript
    import * as vscode from 'vscode';
    import * as lsp from 'vscode-languageclient';

    class CodeCompletionProvider implements vscode.InlineCompletionItemProvider {

        async provideInlineCompletionItems(
            document: vscode.TextDocument,
            position: vscode.Position,
            context: vscode.InlineCompletionContext,
            token: vscode.CancellationToken
        ): Promise<vscode.InlineCompletionList> {

            // Extract context using LSP
            const extractedContext = await this.extractContext(document, position);

            // Call completion API
            const suggestions = await this.completionClient.getCompletions({
                context: extractedContext,
                cursor_position: document.offsetAt(position),
                language: document.languageId,
                stream: false
            });

            return suggestions.map(s => new vscode.InlineCompletionItem(s.text));
        }

        async extractContext(
            document: vscode.TextDocument,
            position: vscode.Position
        ): Promise<CompletionContext> {

            const text = document.getText();
            const offset = document.offsetAt(position);

            // Get imports from LSP
            const imports = await this.getImports(document);

            // Get symbols in scope (functions, classes, variables)
            const symbols = await this.getSymbolsInScope(document, position);

            // Get recent edits from edit history
            const recentEdits = this.editHistory.getRecent(document.uri, 5);

            // Get related files (imported by current file)
            const relatedFiles = await this.getRelatedFiles(document, imports);

            // Context window management: keep most relevant 8K tokens
            const contextWindow = this.buildContextWindow({
                currentFile: text,
                cursorOffset: offset,
                imports: imports,
                symbols: symbols,
                relatedFiles: relatedFiles,
                recentEdits: recentEdits,
                maxTokens: 8000
            });

            return contextWindow;
        }

        async getImports(document: vscode.TextDocument): Promise<Import[]> {
            // Use LSP to get imports
            const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                'vscode.executeDocumentSymbolProvider',
                document.uri
            );

            return symbols
                .filter(s => s.kind === vscode.SymbolKind.Module)
                .map(s => ({
                    module: s.name,
                    line: s.range.start.line,
                    path: this.resolveImportPath(s.name)
                }));
        }

        async getSymbolsInScope(
            document: vscode.TextDocument,
            position: vscode.Position
        ): Promise<Symbol[]> {
            // Get symbols visible at cursor position
            const allSymbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                'vscode.executeDocumentSymbolProvider',
                document.uri
            );

            // Filter to symbols in scope (above cursor or in parent scope)
            return allSymbols.filter(s =>
                s.range.start.line <= position.line &&
                (s.range.end.line > position.line || s.kind === vscode.SymbolKind.Function)
            );
        }

        buildContextWindow(params: ContextParams): CompletionContext {
            const { currentFile, cursorOffset, imports, symbols, relatedFiles, recentEdits, maxTokens } = params;

            // Priority-based token allocation
            let tokens = 0;
            const context: CompletionContext = {
                imports: [],
                symbols: [],
                relatedFiles: [],
                recentEdits: [],
                content: ''
            };

            // 1. Current file around cursor (highest priority): 4000 tokens
            const beforeCursor = currentFile.substring(Math.max(0, cursorOffset - 2000), cursorOffset);
            const afterCursor = currentFile.substring(cursorOffset, Math.min(currentFile.length, cursorOffset + 500));
            context.content = beforeCursor + '|CURSOR|' + afterCursor;
            tokens += this.estimateTokens(context.content);

            // 2. Imports (medium priority): 500 tokens
            context.imports = imports.slice(0, 20);
            tokens += this.estimateTokens(JSON.stringify(context.imports));

            // 3. Symbols in scope (medium priority): 1000 tokens
            context.symbols = symbols.slice(0, 30);
            tokens += this.estimateTokens(JSON.stringify(context.symbols));

            // 4. Recent edits (low priority): 500 tokens
            context.recentEdits = recentEdits.slice(0, 5);
            tokens += this.estimateTokens(JSON.stringify(context.recentEdits));

            // 5. Related files (lowest priority): remaining tokens
            const remainingTokens = maxTokens - tokens;
            for (const file of relatedFiles) {
                const fileTokens = this.estimateTokens(file.content);
                if (tokens + fileTokens > maxTokens) break;
                context.relatedFiles.push(file);
                tokens += fileTokens;
            }

            return context;
        }

        estimateTokens(text: string): number {
            // Rough estimate: 4 chars per token
            return Math.ceil(text.length / 4);
        }
    }
    ```

    ### AST Parsing for Context (Python Example)

    ```python
    import ast
    from typing import List, Dict, Set

    class ContextExtractor:
        def __init__(self):
            self.ast_cache = {}  # Cache parsed ASTs

        def extract_context(self, file_path: str, cursor_line: int, cursor_col: int) -> Dict:
            """Extract context from Python file using AST"""

            with open(file_path, 'r') as f:
                source = f.read()

            # Parse AST (cached)
            if file_path not in self.ast_cache:
                tree = ast.parse(source)
                self.ast_cache[file_path] = tree
            else:
                tree = self.ast_cache[file_path]

            # Extract imports
            imports = self._extract_imports(tree)

            # Extract function/class definitions before cursor
            definitions = self._extract_definitions(tree, cursor_line)

            # Extract variables in scope
            variables = self._extract_variables_in_scope(tree, cursor_line, cursor_col)

            # Find containing function/class
            containing_scope = self._find_containing_scope(tree, cursor_line)

            return {
                'imports': imports,
                'definitions': definitions,
                'variables': variables,
                'containing_scope': containing_scope,
                'source': source
            }

        def _extract_imports(self, tree: ast.AST) -> List[Dict]:
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            'module': f"{node.module}.{alias.name}" if node.module else alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
            return imports

        def _extract_definitions(self, tree: ast.AST, cursor_line: int) -> List[Dict]:
            definitions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.lineno < cursor_line:
                        definitions.append({
                            'name': node.name,
                            'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                            'line': node.lineno,
                            'signature': self._get_function_signature(node) if isinstance(node, ast.FunctionDef) else None
                        })
            return definitions

        def _get_function_signature(self, node: ast.FunctionDef) -> str:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            return_type = ""
            if node.returns:
                return_type = f" -> {ast.unparse(node.returns)}"

            return f"def {node.name}({', '.join(args)}){return_type}"

        def _find_containing_scope(self, tree: ast.AST, cursor_line: int) -> Dict:
            """Find the function/class containing the cursor"""
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.lineno <= cursor_line <= node.end_lineno:
                        return {
                            'name': node.name,
                            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                            'start_line': node.lineno,
                            'end_line': node.end_lineno
                        }
            return None
    ```

    ---

    ## 2. Fill-in-the-Middle (FIM) Prompt Construction

    **Challenge:** Complete code within blocks, not just at end-of-line.

    **Solution:** Use FIM formatting with special tokens.

    ### FIM Prompt Template

    ```python
    class FIMPromptBuilder:
        def __init__(self, model_type: str):
            # Different models use different FIM tokens
            self.model_type = model_type
            self.fim_tokens = self._get_fim_tokens(model_type)

        def _get_fim_tokens(self, model_type: str) -> Dict[str, str]:
            """Get FIM tokens based on model type"""
            if model_type == "codegen":
                return {
                    'prefix': '<fim_prefix>',
                    'suffix': '<fim_suffix>',
                    'middle': '<fim_middle>'
                }
            elif model_type == "starcoder":
                return {
                    'prefix': '<fim_prefix>',
                    'suffix': '<fim_suffix>',
                    'middle': '<fim_middle>'
                }
            elif model_type == "code-llama":
                return {
                    'prefix': '<PRE>',
                    'suffix': '<SUF>',
                    'middle': '<MID>'
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        def build_fim_prompt(
            self,
            prefix: str,
            suffix: str,
            language: str,
            max_prefix_tokens: int = 2000,
            max_suffix_tokens: int = 500
        ) -> str:
            """
            Build FIM prompt for code completion.

            Example:
            prefix: "def calculate_sum(a, b):\n    "
            suffix: "\n    return result"
            → Model predicts: "result = a + b"
            """

            # Truncate prefix to max tokens (keep end of prefix)
            prefix_tokens = self._tokenize(prefix)
            if len(prefix_tokens) > max_prefix_tokens:
                prefix = self._detokenize(prefix_tokens[-max_prefix_tokens:])

            # Truncate suffix to max tokens (keep start of suffix)
            suffix_tokens = self._tokenize(suffix)
            if len(suffix_tokens) > max_suffix_tokens:
                suffix = self._detokenize(suffix_tokens[:max_suffix_tokens])

            # Add language hint
            language_hint = f"# Language: {language}\n" if language else ""

            # Construct FIM prompt
            prompt = (
                f"{language_hint}"
                f"{self.fim_tokens['prefix']}"
                f"{prefix}"
                f"{self.fim_tokens['suffix']}"
                f"{suffix}"
                f"{self.fim_tokens['middle']}"
            )

            return prompt

        def build_chat_prompt(
            self,
            user_message: str,
            code_context: Dict,
            conversation_history: List[Dict]
        ) -> str:
            """Build chat prompt with code context"""

            system_prompt = """You are an expert programmer assistant.
You help developers write, understand, and improve code.
Always provide clear, concise, and correct code examples.
"""

            # Add code context
            context_str = ""
            if code_context.get('file_path'):
                context_str += f"\n# Current file: {code_context['file_path']}\n"

            if code_context.get('selected_code'):
                context_str += f"\n# Selected code:\n```{code_context.get('language', '')}\n{code_context['selected_code']}\n```\n"

            if code_context.get('imports'):
                context_str += f"\n# Imports:\n" + "\n".join(code_context['imports']) + "\n"

            # Build conversation history
            conversation = f"<system>{system_prompt}</system>\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg['role']
                content = msg['content']
                conversation += f"<{role}>{content}</{role}>\n"

            # Add context and user message
            conversation += f"{context_str}\n<user>{user_message}</user>\n<assistant>"

            return conversation
    ```

    ---

    ## 3. Prompt Prefix Caching & KV Cache Optimization

    **Challenge:** Repeated processing of same file context wastes GPU time.

    **Solution:** Cache KV (Key-Value) tensors for prompt prefixes, only process new tokens.

    ### KV Cache Implementation

    ```python
    import hashlib
    import torch
    from typing import Optional, Tuple

    class PromptPrefixCache:
        def __init__(self, redis_client, s3_client, gpu_memory_limit_gb: int = 40):
            self.redis = redis_client
            self.s3 = s3_client
            self.gpu_cache = {}  # In-memory GPU cache
            self.gpu_memory_limit = gpu_memory_limit_gb * 1024**3  # bytes
            self.current_gpu_memory = 0

        def get_cache_key(self, prefix: str, model_version: str) -> str:
            """Generate cache key from prefix"""
            prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
            return f"kv_cache:{model_version}:{prefix_hash}"

        def get_kv_cache(
            self,
            prefix: str,
            model_version: str
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            """
            Get cached KV tensors for prefix.
            Returns (key_cache, value_cache) or None if miss.
            """
            cache_key = self.get_cache_key(prefix, model_version)

            # Check GPU memory cache first (hot cache)
            if cache_key in self.gpu_cache:
                print(f"GPU cache HIT: {cache_key}")
                self._update_access_time(cache_key)
                return self.gpu_cache[cache_key]

            # Check Redis metadata
            metadata = self.redis.get(cache_key)
            if not metadata:
                print(f"Cache MISS: {cache_key}")
                return None

            metadata = json.loads(metadata)

            # Load KV cache from S3/SSD
            s3_path = metadata['s3_path']
            kv_cache = self._load_from_s3(s3_path)

            # Load into GPU memory (evict if needed)
            if self.current_gpu_memory + kv_cache['size'] > self.gpu_memory_limit:
                self._evict_lru()

            self.gpu_cache[cache_key] = (kv_cache['keys'], kv_cache['values'])
            self.current_gpu_memory += kv_cache['size']

            print(f"Loaded KV cache from S3: {cache_key}")
            return (kv_cache['keys'], kv_cache['values'])

        def save_kv_cache(
            self,
            prefix: str,
            model_version: str,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            ttl_seconds: int = 86400
        ):
            """Save KV cache to S3 and Redis metadata"""
            cache_key = self.get_cache_key(prefix, model_version)

            # Save to S3
            s3_path = f"kv_cache/{model_version}/{cache_key}.pt"
            self._save_to_s3(s3_path, {
                'keys': key_cache.cpu(),
                'values': value_cache.cpu(),
                'size': key_cache.element_size() * key_cache.numel() + value_cache.element_size() * value_cache.numel()
            })

            # Save metadata to Redis
            metadata = {
                's3_path': s3_path,
                'model_version': model_version,
                'token_count': key_cache.shape[1],  # Sequence length
                'created_at': time.time(),
                'access_count': 0
            }
            self.redis.setex(cache_key, ttl_seconds, json.dumps(metadata))

            # Add to GPU cache
            self.gpu_cache[cache_key] = (key_cache, value_cache)
            self.current_gpu_memory += metadata['size']

        def _evict_lru(self):
            """Evict least recently used cache entries"""
            # Sort by access time, evict oldest
            sorted_keys = sorted(self.gpu_cache.keys(), key=lambda k: self._get_access_time(k))

            while self.current_gpu_memory > self.gpu_memory_limit * 0.8 and sorted_keys:
                evict_key = sorted_keys.pop(0)
                kv_cache = self.gpu_cache[evict_key]
                size = kv_cache[0].element_size() * kv_cache[0].numel() + kv_cache[1].element_size() * kv_cache[1].numel()

                del self.gpu_cache[evict_key]
                self.current_gpu_memory -= size
                print(f"Evicted KV cache: {evict_key}")
    ```

    ### Inference with KV Cache Reuse

    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class OptimizedInferenceEngine:
        def __init__(self, model_path: str, cache_manager: PromptPrefixCache):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.cache_manager = cache_manager

        def generate_completion(
            self,
            prefix: str,
            suffix: str,
            max_new_tokens: int = 100,
            temperature: float = 0.2
        ) -> str:
            """
            Generate completion with KV cache reuse.

            Steps:
            1. Check if prefix is cached
            2. If cached, only tokenize new tokens since cache point
            3. Reuse KV cache, only compute new tokens
            4. 3-5x speedup for cached prefixes
            """

            # Build FIM prompt
            full_prompt = self.build_fim_prompt(prefix, suffix)

            # Check cache for prefix
            cached_kv = self.cache_manager.get_kv_cache(prefix, self.model.config.model_version)

            if cached_kv is not None:
                # Cache HIT: Only tokenize new tokens
                key_cache, value_cache = cached_kv

                # Find cache boundary (where cached prefix ends)
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                cached_length = key_cache.shape[1]  # Sequence length in cache

                # Tokenize only new part (suffix + middle token)
                new_prompt = self.build_fim_prompt(prefix[-100:], suffix)  # Overlap for safety
                new_tokens = self.tokenizer.encode(new_prompt, return_tensors="pt").to(self.model.device)

                # Run inference with cached KV
                outputs = self.model.generate(
                    input_ids=new_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    past_key_values=cached_kv,  # Reuse cached KV
                    use_cache=True
                )

                print(f"✓ KV cache reused, saved {cached_length} token computations")

            else:
                # Cache MISS: Full inference
                input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

                # Generate with cache recording
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # Extract and save KV cache for future use
                if hasattr(outputs, 'past_key_values'):
                    past_kv = outputs.past_key_values
                    # Save prefix KV cache (first N layers)
                    self.cache_manager.save_kv_cache(
                        prefix=prefix,
                        model_version=self.model.config.model_version,
                        key_cache=past_kv[0][0],  # First layer key cache
                        value_cache=past_kv[0][1]  # First layer value cache
                    )

                print(f"✗ Cache miss, computed {input_ids.shape[1]} tokens")

            # Decode output
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completion
    ```

    **Cache Hit Rate Impact:**

    | Scenario | Cache Hit Rate | Avg Latency | GPU Utilization |
    |----------|----------------|-------------|-----------------|
    | No caching | 0% | 250ms | 60% |
    | Prefix caching | 70% | 120ms (52% reduction) | 85% |
    | Full completion caching | 30% | 5ms (cache hit), 250ms (miss) | 70% |
    | **Combined (both layers)** | **85% effective** | **80ms average** | **88%** |

    ---

    ## 4. Low-Latency Inference Optimization

    **Challenge:** <300ms latency requirement for inline completions.

    **Solution:** Multiple optimization techniques.

    ### Techniques

    | Technique | Description | Latency Improvement | Trade-off |
    |-----------|-------------|---------------------|-----------|
    | **Speculative Decoding** | Use small "draft" model to predict multiple tokens, verify with main model | 2-3x faster | 5-10% quality loss |
    | **Continuous Batching** | Dynamic batching of requests (not fixed batch size) | 3-5x throughput | Slightly increased latency for small batches |
    | **Flash Attention** | Optimized attention implementation (less GPU memory, faster) | 2x faster attention | Requires recent GPUs (A100+) |
    | **Quantization (INT8)** | Reduce precision from FP16 to INT8 | 2x faster, 50% less memory | 1-2% quality loss |
    | **Prompt Caching** | Reuse KV cache for repeated prefixes | 3-5x faster (cache hit) | Requires cache infrastructure |
    | **Model Distillation** | Train smaller model from larger (1B from 7B) | 10x faster | 10-20% quality loss |

    ### Speculative Decoding Implementation

    ```python
    import torch
    from transformers import AutoModelForCausalLM

    class SpeculativeDecoder:
        """
        Speculative decoding: Use small draft model to predict tokens,
        verify with main model in parallel.

        Example:
        - Draft model (100M params): Predicts 5 tokens in 20ms
        - Main model (1B params): Verifies 5 tokens in 30ms
        - Total: 50ms vs 150ms (5 × 30ms) = 3x speedup
        """

        def __init__(self, draft_model_path: str, main_model_path: str):
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.main_model = AutoModelForCausalLM.from_pretrained(
                main_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.draft_model.eval()
            self.main_model.eval()

        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 50,
            gamma: int = 5,  # Number of draft tokens per iteration
            temperature: float = 0.2
        ) -> torch.Tensor:
            """
            Generate tokens using speculative decoding.

            Args:
                input_ids: Input token IDs [batch_size, seq_len]
                max_new_tokens: Maximum tokens to generate
                gamma: Number of tokens to draft per iteration
                temperature: Sampling temperature

            Returns:
                Generated token IDs
            """

            generated_tokens = input_ids.clone()
            n_generated = 0
            n_accepted_total = 0
            n_drafted_total = 0

            while n_generated < max_new_tokens:
                # Step 1: Draft model generates gamma tokens quickly
                draft_tokens = []
                draft_logits = []
                current_input = generated_tokens

                for _ in range(gamma):
                    draft_output = self.draft_model(current_input)
                    draft_logit = draft_output.logits[:, -1, :]
                    draft_logits.append(draft_logit)

                    # Sample next token
                    draft_token = self._sample(draft_logit, temperature)
                    draft_tokens.append(draft_token)

                    # Append to input for next iteration
                    current_input = torch.cat([current_input, draft_token], dim=1)

                draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch_size, gamma]
                n_drafted_total += gamma

                # Step 2: Main model verifies all draft tokens in parallel
                verify_input = torch.cat([generated_tokens, draft_tokens], dim=1)
                main_output = self.main_model(verify_input)
                main_logits = main_output.logits[:, -gamma-1:-1, :]  # Logits for each draft position

                # Step 3: Accept/reject each draft token
                accepted_tokens = []
                for i in range(gamma):
                    draft_token = draft_tokens[:, i:i+1]
                    main_logit = main_logits[:, i, :]
                    draft_logit = draft_logits[i]

                    # Acceptance probability: min(1, p_main(token) / p_draft(token))
                    main_prob = torch.softmax(main_logit / temperature, dim=-1)
                    draft_prob = torch.softmax(draft_logit / temperature, dim=-1)

                    token_idx = draft_token.item()
                    accept_prob = min(1.0, (main_prob[0, token_idx] / (draft_prob[0, token_idx] + 1e-10)).item())

                    # Accept or reject
                    if torch.rand(1).item() < accept_prob:
                        accepted_tokens.append(draft_token)
                    else:
                        # Reject: resample from main model distribution
                        corrected_token = self._sample(main_logit, temperature)
                        accepted_tokens.append(corrected_token)
                        break  # Stop accepting after first rejection

                # Step 4: Append accepted tokens
                if accepted_tokens:
                    accepted = torch.cat(accepted_tokens, dim=1)
                    generated_tokens = torch.cat([generated_tokens, accepted], dim=1)
                    n_generated += accepted.shape[1]
                    n_accepted_total += accepted.shape[1]

                # Early exit if no tokens accepted (rare)
                if not accepted_tokens:
                    break

            acceptance_rate = n_accepted_total / n_drafted_total if n_drafted_total > 0 else 0
            print(f"Speculative decoding: {n_accepted_total}/{n_drafted_total} accepted ({acceptance_rate:.1%})")

            return generated_tokens

        def _sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
            """Sample token from logits with temperature"""
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
    ```

    **Latency Breakdown (With Optimizations):**

    ```
    Total latency for completion: 180ms

    1. Network (IDE → API): 20ms
    2. Context extraction: 15ms (AST parsing, cached)
    3. Cache lookup: 5ms (Redis check)
    4. Prompt construction: 10ms
    5. Model inference: 100ms
       - Tokenization: 5ms
       - KV cache load (70% hit): 10ms
       - Forward pass (speculative): 75ms (draft 20ms + verify 55ms)
       - Decoding: 10ms
    6. Post-processing (security scan): 10ms
    7. Network (API → IDE): 20ms

    Critical path optimizations:
    - Prompt caching: 100ms → 30ms (cache hit)
    - Speculative decoding: 250ms → 75ms (3x speedup)
    - Continuous batching: +20% throughput
    ```

    ---

    ## 5. Security: Credential Scanning

    **Challenge:** Prevent AI from suggesting code with leaked secrets (API keys, passwords).

    **Solution:** Multi-layer scanning (regex + ML classifier).

    ### Credential Scanner Implementation

    ```python
    import re
    from typing import List, Tuple, Optional
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class CredentialScanner:
        def __init__(self, ml_model_path: Optional[str] = None):
            # Regex patterns for common secrets
            self.patterns = {
                'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
                'aws_secret_key': re.compile(r'aws_secret_access_key\s*=\s*[\'"][A-Za-z0-9/+=]{40}[\'"]', re.IGNORECASE),
                'github_token': re.compile(r'ghp_[a-zA-Z0-9]{36}'),
                'slack_token': re.compile(r'xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[a-zA-Z0-9]{24,32}'),
                'stripe_key': re.compile(r'sk_live_[a-zA-Z0-9]{24}'),
                'generic_api_key': re.compile(r'api[_-]?key\s*[=:]\s*[\'"][a-zA-Z0-9_\-]{20,}[\'"]', re.IGNORECASE),
                'password': re.compile(r'password\s*[=:]\s*[\'"][^\'"]{8,}[\'"]', re.IGNORECASE),
                'private_key': re.compile(r'-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----'),
                'jwt_token': re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
            }

            # ML-based classifier for complex patterns
            if ml_model_path:
                self.ml_model = AutoModelForSequenceClassification.from_pretrained(ml_model_path)
                self.ml_tokenizer = AutoTokenizer.from_pretrained(ml_model_path)
                self.ml_model.eval()
            else:
                self.ml_model = None

        def scan(self, code: str) -> Tuple[bool, List[Dict]]:
            """
            Scan code for credentials.

            Returns:
                (is_safe, detected_secrets)
            """
            detected = []

            # Step 1: Regex-based detection (fast, high precision)
            for secret_type, pattern in self.patterns.items():
                matches = pattern.finditer(code)
                for match in matches:
                    detected.append({
                        'type': secret_type,
                        'value': self._redact(match.group(0)),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })

            # Step 2: ML-based detection (slower, catches complex patterns)
            if self.ml_model and not detected:
                ml_result = self._ml_scan(code)
                if ml_result['is_secret']:
                    detected.append({
                        'type': 'potential_secret',
                        'confidence': ml_result['confidence'],
                        'method': 'ml'
                    })

            # Step 3: Entropy analysis (detect high-entropy strings)
            if not detected:
                high_entropy_strings = self._find_high_entropy_strings(code)
                for s in high_entropy_strings:
                    detected.append({
                        'type': 'high_entropy',
                        'value': self._redact(s),
                        'confidence': 0.7,
                        'method': 'entropy'
                    })

            is_safe = len(detected) == 0
            return is_safe, detected

        def _ml_scan(self, code: str) -> Dict:
            """Use ML model to detect secrets"""
            inputs = self.ml_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Binary classification: [not_secret, secret]
                secret_prob = probs[0, 1].item()

                return {
                    'is_secret': secret_prob > 0.8,
                    'confidence': secret_prob
                }

        def _find_high_entropy_strings(self, code: str, min_length: int = 20) -> List[str]:
            """Find strings with high Shannon entropy (likely secrets)"""
            import math
            from collections import Counter

            def entropy(s: str) -> float:
                if not s:
                    return 0
                counter = Counter(s)
                length = len(s)
                return -sum((count/length) * math.log2(count/length) for count in counter.values())

            # Extract string literals
            string_pattern = re.compile(r'[\'"]([^\'"]{20,})[\'"]')
            strings = string_pattern.findall(code)

            high_entropy = []
            for s in strings:
                if len(s) >= min_length:
                    ent = entropy(s)
                    # High entropy threshold: 4.5+ (random-looking strings)
                    if ent > 4.5:
                        high_entropy.append(s)

            return high_entropy

        def _redact(self, value: str) -> str:
            """Redact secret value for logging"""
            if len(value) <= 8:
                return '***'
            return value[:4] + '***' + value[-4:]

    # Usage in completion pipeline
    def post_process_completion(completion: str) -> Tuple[str, bool]:
        scanner = CredentialScanner()
        is_safe, secrets = scanner.scan(completion)

        if not is_safe:
            print(f"⚠️ Blocked completion with {len(secrets)} potential secrets")
            for secret in secrets:
                print(f"  - {secret['type']} (confidence: {secret['confidence']:.2f})")
            return None, False

        return completion, True
    ```

    **Real-World Example (Blocked Completion):**

    ```python
    # User prompt: "Write code to upload to S3"
    # AI draft completion:
    def upload_to_s3(file_path):
        s3_client = boto3.client(
            's3',
            aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # ❌ BLOCKED
            aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'  # ❌ BLOCKED
        )
        s3_client.upload_file(file_path, 'my-bucket', 'file.txt')

    # Scanner detects AWS credentials → Block suggestion
    # Alternative safe suggestion:
    def upload_to_s3(file_path):
        # Use environment variables or IAM roles for credentials
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, 'my-bucket', 'file.txt')
    ```

=== "📈 Step 6: Scalability & Performance"

    ## Scalability Considerations

    ### 1. Horizontal Scaling (GPU Inference)

    **Challenge:** Handle 230K peak completion requests/sec.

    **Solution:** Auto-scaling GPU pools with load balancing.

    ```python
    class GPULoadBalancer:
        """
        Intelligent load balancing across GPU replicas.

        Routing strategies:
        1. Least-loaded: Route to GPU with smallest queue
        2. Round-robin: Distribute evenly
        3. Affinity: Route same user to same GPU (cache locality)
        """

        def __init__(self, gpu_endpoints: List[str]):
            self.endpoints = gpu_endpoints
            self.queue_sizes = {ep: 0 for ep in gpu_endpoints}
            self.user_affinity = {}  # user_id → endpoint

        def route_request(self, request: CompletionRequest) -> str:
            """
            Route completion request to optimal GPU.

            Priority:
            1. If prefix is cached on specific GPU → route there (cache locality)
            2. Else route to least-loaded GPU
            """

            # Check cache affinity
            cache_key = self._get_cache_key(request.context)
            if cache_key in self.user_affinity:
                endpoint = self.user_affinity[cache_key]
                if self.queue_sizes[endpoint] < 100:  # Not overloaded
                    return endpoint

            # Route to least-loaded GPU
            endpoint = min(self.queue_sizes.items(), key=lambda x: x[1])[0]

            # Update affinity
            self.user_affinity[cache_key] = endpoint

            return endpoint

        def update_queue_size(self, endpoint: str, size: int):
            """Update queue size from GPU worker heartbeat"""
            self.queue_sizes[endpoint] = size
    ```

    **Auto-scaling Policy:**

    ```yaml
    # Kubernetes HPA (Horizontal Pod Autoscaler)
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: completion-model-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: completion-model-gpu
      minReplicas: 50
      maxReplicas: 500
      metrics:
      - type: Pods
        pods:
          metric:
            name: gpu_utilization
          target:
            type: AverageValue
            averageValue: "80"  # Target 80% GPU utilization
      - type: Pods
        pods:
          metric:
            name: queue_length
          target:
            type: AverageValue
            averageValue: "32"  # Target 32 requests in queue
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60
          policies:
          - type: Percent
            value: 50  # Scale up by 50% per minute
            periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
          - type: Pods
            value: 10  # Scale down by 10 pods per 5 minutes
            periodSeconds: 300
    ```

    ---

    ### 2. Caching Strategy (Multi-tier)

    **Cache Hierarchy:**

    ```
    L1: Completion Cache (Redis, exact match)
        - Hit rate: 30%
        - Latency: 5ms
        - Size: 200 GB
        - TTL: 1 hour

    L2: Prompt Prefix Cache (Redis + S3, partial match)
        - Hit rate: 70% (of L1 misses)
        - Latency: 50ms (load KV cache from S3)
        - Size: 1 TB (metadata in Redis, tensors in S3)
        - TTL: 24 hours

    L3: Model KV Cache (GPU memory, token-level)
        - Hit rate: 100% (for repeated tokens in same request)
        - Latency: 0ms (in GPU memory)
        - Size: 250 TB (distributed across GPUs)
        - TTL: Request lifetime

    Effective cache hit rate:
      L1 (30%) + L2 (70% × 70%) + L3 (100% within request) = 79% effective
    ```

    **Cache Invalidation Strategy:**

    ```python
    class CacheInvalidator:
        """
        Invalidate cache when code changes.

        Challenges:
        - File edited → invalidate cache for that file
        - File renamed → invalidate cache for old path
        - Project-wide refactor → invalidate all caches
        """

        def __init__(self, redis_client):
            self.redis = redis_client

        def on_file_edit(self, file_path: str, user_id: str):
            """Invalidate cache when file is edited"""
            # Invalidate completion cache for this file
            pattern = f"completion:*:{file_path}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                print(f"Invalidated {len(keys)} completion cache entries for {file_path}")

            # Invalidate prefix cache
            prefix_pattern = f"prefix:*:{file_path}:*"
            prefix_keys = self.redis.keys(prefix_pattern)
            if prefix_keys:
                self.redis.delete(*prefix_keys)
                print(f"Invalidated {len(prefix_keys)} prefix cache entries")

        def on_file_rename(self, old_path: str, new_path: str, user_id: str):
            """Handle file rename"""
            self.on_file_edit(old_path, user_id)
            self.on_file_edit(new_path, user_id)

        def on_project_refactor(self, user_id: str):
            """Invalidate all caches for user (rare, expensive)"""
            pattern = f"*:{user_id}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                print(f"⚠️ Invalidated ALL cache for user {user_id} ({len(keys)} entries)")
    ```

    ---

    ### 3. Database Sharding (Logs & Analytics)

    **Challenge:** 10B completion logs/day → 5 TB/day.

    **Solution:** Shard ClickHouse by date + user_id.

    ```sql
    -- ClickHouse sharding configuration
    CREATE TABLE completions_log_distributed AS completions_log
    ENGINE = Distributed(
        cluster_logs,  -- Cluster name
        default,       -- Database
        completions_log_local,  -- Local table on each shard
        rand()         -- Sharding key (random distribution)
    );

    -- Local table on each shard (16 shards)
    CREATE TABLE completions_log_local ON CLUSTER cluster_logs (
        completion_id String,
        user_id String,
        timestamp DateTime64(3),
        language String,
        latency_ms UInt32,
        accepted Boolean,
        date Date MATERIALIZED toDate(timestamp)
    )
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/completions_log', '{replica}')
    PARTITION BY toYYYYMM(date)
    ORDER BY (date, user_id, timestamp)
    TTL date + INTERVAL 30 DAY;

    -- Query across all shards
    SELECT
        language,
        countIf(accepted = true) / count() as acceptance_rate,
        avg(latency_ms) as avg_latency
    FROM completions_log_distributed
    WHERE date >= today() - 7
    GROUP BY language;
    ```

    ---

    ### 4. Rate Limiting

    **Strategy:** Token bucket per user tier.

    | Tier | Completions/min | Chat messages/min | Burst capacity |
    |------|-----------------|-------------------|----------------|
    | Free | 100 | 10 | 150 |
    | Pro | 500 | 50 | 750 |
    | Team | 2000 | 200 | 3000 |
    | Enterprise | Unlimited | Unlimited | N/A |

    ```python
    import time
    from typing import Optional

    class TokenBucketRateLimiter:
        def __init__(self, redis_client):
            self.redis = redis_client

        def check_rate_limit(
            self,
            user_id: str,
            tier: str,
            request_type: str  # 'completion' or 'chat'
        ) -> Tuple[bool, Optional[int]]:
            """
            Check if user can make request.

            Returns:
                (allowed, retry_after_seconds)
            """
            config = self._get_tier_config(tier, request_type)
            key = f"rate_limit:{user_id}:{request_type}"

            # Get current bucket state
            bucket = self.redis.get(key)
            now = time.time()

            if bucket:
                bucket = json.loads(bucket)
                tokens = bucket['tokens']
                last_refill = bucket['last_refill']
            else:
                # Initialize bucket
                tokens = config['bucket_size']
                last_refill = now

            # Refill tokens based on time elapsed
            elapsed = now - last_refill
            refill_amount = elapsed * (config['refill_rate'] / 60.0)  # per second
            tokens = min(config['bucket_size'], tokens + refill_amount)

            # Check if request can be made
            if tokens >= 1.0:
                # Allow request
                tokens -= 1.0
                bucket_state = {
                    'tokens': tokens,
                    'last_refill': now
                }
                self.redis.setex(key, 3600, json.dumps(bucket_state))
                return True, None
            else:
                # Rate limited
                retry_after = int((1.0 - tokens) / (config['refill_rate'] / 60.0))
                return False, retry_after

        def _get_tier_config(self, tier: str, request_type: str) -> Dict:
            configs = {
                ('free', 'completion'): {'bucket_size': 150, 'refill_rate': 100},
                ('free', 'chat'): {'bucket_size': 15, 'refill_rate': 10},
                ('pro', 'completion'): {'bucket_size': 750, 'refill_rate': 500},
                ('pro', 'chat'): {'bucket_size': 75, 'refill_rate': 50},
                ('team', 'completion'): {'bucket_size': 3000, 'refill_rate': 2000},
                ('team', 'chat'): {'bucket_size': 300, 'refill_rate': 200},
            }
            return configs.get((tier, request_type), configs[('free', request_type)])
    ```

    ---

    ### 5. Monitoring & Observability

    **Key Metrics:**

    ```python
    # Prometheus metrics
    from prometheus_client import Counter, Histogram, Gauge

    # Completion metrics
    completion_requests_total = Counter(
        'completion_requests_total',
        'Total completion requests',
        ['language', 'model', 'cache_hit']
    )

    completion_latency = Histogram(
        'completion_latency_seconds',
        'Completion latency',
        ['language', 'cache_hit'],
        buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    )

    completion_acceptance_rate = Gauge(
        'completion_acceptance_rate',
        'Percentage of completions accepted',
        ['language', 'model']
    )

    # GPU metrics
    gpu_utilization = Gauge(
        'gpu_utilization_percent',
        'GPU utilization percentage',
        ['gpu_id', 'model']
    )

    gpu_memory_used = Gauge(
        'gpu_memory_used_bytes',
        'GPU memory used',
        ['gpu_id']
    )

    inference_queue_size = Gauge(
        'inference_queue_size',
        'Number of requests in inference queue',
        ['model']
    )

    # Cache metrics
    cache_hit_rate = Gauge(
        'cache_hit_rate',
        'Cache hit rate',
        ['cache_type']  # 'completion', 'prefix', 'kv'
    )

    # Security metrics
    credential_detections = Counter(
        'credential_detections_total',
        'Number of completions blocked for credentials',
        ['secret_type']
    )
    ```

    **Dashboards (Grafana):**

    ```
    1. Completion Performance Dashboard:
       - P50/P95/P99 latency (by language, cache hit/miss)
       - Requests per second
       - Error rate
       - Cache hit rate (L1/L2/L3)

    2. GPU Utilization Dashboard:
       - GPU utilization % (by model, GPU ID)
       - GPU memory usage
       - Queue length
       - Batch size
       - Tokens per second

    3. User Experience Dashboard:
       - Acceptance rate (by language, model)
       - Time to first token (TTFT)
       - Multi-line completion rate
       - Completion length distribution

    4. Security Dashboard:
       - Credential detections (by type)
       - Blocked completions rate
       - False positive rate
    ```

=== "🎯 Step 7: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5 minutes)

    **Questions to ask:**

    - **Scope:** "Is this inline code completion only, or chat-based generation too?"
    - **Scale:** "How many developers? How many completions per day?"
    - **Latency:** "What's the latency target for inline completions? Chat?"
    - **Languages:** "All programming languages or specific ones?"
    - **IDE support:** "VSCode only, or JetBrains, Vim, etc.?"
    - **Model:** "Do we need to train models, or assume pre-trained?"
    - **Security:** "Any requirements around credential scanning or privacy?"
    - **Context:** "How much context should we extract? Entire project or just current file?"

    ---

    ### 2. High-Level Design (15 minutes)

    **Start with simple diagram:**

    ```
    [IDE Extension] → [API Gateway] → [Completion Service] → [LLM Inference] → Response
                                          ↓
                                     [Cache Layer]
    ```

    **Then expand:**

    - IDE integration (LSP)
    - Context extraction service
    - Multi-tier caching (completion, prefix, KV)
    - GPU inference cluster
    - Security scanning
    - Telemetry

    ---

    ### 3. Deep Dive (30 minutes)

    **Pick 2-3 areas to dive deep based on interviewer interest:**

    **A. Context Extraction:**
    - LSP integration for IDE
    - AST parsing for imports, definitions
    - Multi-file context (how to select related files?)
    - Context window management (8K tokens)

    **B. Low-Latency Inference:**
    - Small model (1B params) for speed
    - Speculative decoding (2-3x speedup)
    - Prompt prefix caching (70% hit rate)
    - KV cache reuse
    - Continuous batching

    **C. Caching Strategy:**
    - L1: Completion cache (exact match, 30% hit)
    - L2: Prefix cache (partial match, 70% hit)
    - L3: KV cache (token-level, in GPU memory)
    - Cache invalidation on file edits

    **D. Security:**
    - Credential scanning (regex + ML)
    - High-entropy string detection
    - Blocked pattern examples (AWS keys, GitHub tokens)

    ---

    ### 4. Capacity Estimation (10 minutes)

    **Walk through key numbers:**

    ```
    Users: 50M DAU
    Completions: 200/user/day = 10B/day = 115K req/sec
    Tokens: 2K context + 100 output = 2.1K tokens/request
    GPUs needed: ~1,150 A100 GPUs (with caching & batching)
    Cost: ~$2M/month GPU cost
    Storage: ~300 TB (logs, models, cache)
    ```

    ---

    ### 5. Trade-offs & Follow-ups (10 minutes)

    **Common follow-up questions:**

    ---

    ## Common Follow-up Questions

    ### Q1: How do you handle multi-file context?

    **Answer:**
    ```
    1. Start with current file (highest priority)
    2. Parse imports → identify directly referenced files
    3. Use vector embeddings to find semantically similar code
    4. Rank files by relevance:
       - Imported by current file: Score 1.0
       - Same directory: Score 0.8
       - Semantic similarity: Score 0.5-0.7
    5. Include top N files that fit in context window (8K tokens)
    6. Cache file embeddings to avoid recomputation

    Trade-off: More context = better suggestions, but slower inference
    Solution: Adaptive context size based on request type:
       - Inline completion: 4K tokens (current file only)
       - Chat generation: 16K tokens (multi-file)
    ```

    ---

    ### Q2: How do you improve acceptance rate?

    **Answer:**
    ```
    1. Model quality:
       - Fine-tune on high-quality code (repositories with >100 stars)
       - Filter training data by language, framework
       - Continuous learning from accepted completions (feedback loop)

    2. Ranking multiple candidates:
       - Generate 3-5 completions with different temperatures
       - Score by:
         * Syntactic correctness (AST parsing)
         * Semantic similarity to context (embedding cosine similarity)
         * User's coding style (learned from history)
       - Return highest-scoring suggestion

    3. Personalization:
       - Learn user preferences (variable naming, code style)
       - Adapt to project conventions (tabs vs spaces, camelCase vs snake_case)
       - Store user feedback (accepted/rejected patterns)

    4. Context quality:
       - Better context extraction → better suggestions
       - Include recent edits (what user just typed)
       - Include error messages from IDE (if compiler error, suggest fix)

    Typical acceptance rate progression:
       - Baseline model: 20-25%
       - + Context optimization: 30-35%
       - + Multi-candidate ranking: 35-40%
       - + Personalization: 40-50%
    ```

    ---

    ### Q3: How do you handle rate limiting without hurting UX?

    **Answer:**
    ```
    1. Client-side debouncing:
       - Wait 200-300ms after user stops typing before sending request
       - Cancel in-flight requests if user keeps typing
       - Avoids sending request on every keystroke

    2. Intelligent triggering:
       - Trigger on specific events: newline, opening brace, function call
       - Don't trigger mid-word or on whitespace
       - Reduces unnecessary requests by 50-70%

    3. Tiered rate limits with soft vs hard limits:
       - Soft limit: Warn user, but allow requests
       - Hard limit: Block requests, show "rate limited" message
       - Example: Free tier 100 req/min (soft 80, hard 100)

    4. Priority queue:
       - Inline completions: High priority (user waiting)
       - Chat messages: Medium priority (async)
       - Background indexing: Low priority (best-effort)

    5. Graceful degradation:
       - If rate limited, fall back to simpler completions (templates, snippets)
       - Show cached suggestions instead of new generation
       - Inform user: "Rate limit reached, upgrade for more completions"
    ```

    ---

    ### Q4: How do you handle very large files (>1M lines)?

    **Answer:**
    ```
    Problem: Cannot fit entire file in context window (8K-32K tokens).

    Solution: Sliding window + hierarchical context

    1. Sliding window around cursor:
       - Take 2K tokens before cursor, 500 tokens after
       - Always include cursor position (highest relevance)

    2. Hierarchical context:
       - Level 1: Current function/class (highest priority)
       - Level 2: Imports + function signatures in file
       - Level 3: Related files (imports)
       - Allocate tokens by level (60% / 30% / 10%)

    3. Skeleton extraction:
       - For large files, extract "skeleton":
         * All function/class signatures
         * Import statements
         * Global variables
         * Docstrings
       - Skeleton is much smaller (10x compression)
       - Provides high-level context without full code

    4. Vector search for relevant sections:
       - Pre-index file sections (every 50 lines)
       - Use embedding similarity to find relevant sections
       - Include top-3 most similar sections in context

    Example:
    File: 10,000 lines Python file
    Context budget: 8,000 tokens

    Allocation:
    - Current function (lines 5000-5050): 2,000 tokens (25%)
    - Skeleton (imports, signatures): 3,000 tokens (37.5%)
    - Related files (imported modules): 2,000 tokens (25%)
    - Vector-searched sections: 1,000 tokens (12.5%)

    Result: Comprehensive context despite large file size
    ```

    ---

    ### Q5: How do you prevent model from suggesting copyrighted code?

    **Answer:**
    ```
    Challenge: Training data includes copyrighted code from GitHub.
    Risk: Model may memorize and regurgitate copyrighted code.

    Solutions:

    1. Training data filtering:
       - Remove code from repositories with restrictive licenses (GPL, proprietary)
       - Only train on permissive licenses (MIT, Apache, BSD)
       - Deduplicate training data (remove exact duplicates)

    2. Membership inference detection:
       - Test if model memorized training data
       - If model reproduces >10 consecutive lines, flag as potential memorization
       - Filter out memorized snippets

    3. Post-generation filtering:
       - Check if suggestion matches known code snippets (fingerprinting)
       - Hash suggestion, compare against database of copyrighted code
       - If match, block suggestion or add attribution

    4. Diversity in generation:
       - Use higher temperature (0.3-0.4) to increase randomness
       - Avoid deterministic completions (exact reproductions)
       - Generate multiple candidates, pick most diverse

    5. Legal safeguards:
       - Terms of service: User responsible for checking licenses
       - Indemnification: Offer legal protection for enterprise customers
       - Attribution feature: If code matches known source, show attribution

    Real-world example:
    - GitHub Copilot: Trained on public code, offers "duplicate detection"
    - OpenAI Codex: Filters training data by license, offers copyright guarantee
    ```

    ---

    ### Q6: How do you handle context switching between files?

    **Answer:**
    ```
    Challenge: User switches between files frequently (every 30-60 seconds).
    Need to maintain context across files without latency spikes.

    Solution: Session-based context management

    1. Session cache:
       - Store context for last 5 opened files in Redis
       - Key: session:{user_id}:{file_path}
       - TTL: 1 hour (refreshed on access)
       - Contains: Parsed AST, imports, symbols, recent edits

    2. Pre-warming on file open:
       - When user opens file, async load context in background
       - Parse AST, extract imports, resolve symbols
       - By the time user starts typing, context ready
       - Latency: 0ms (already cached)

    3. Workspace-level caching:
       - Cache imports graph for entire workspace
       - When user switches files, context includes:
         * New file's content
         * Cached workspace imports
         * Previously opened files (if related)
       - Avoids re-parsing on every switch

    4. Context prefetching:
       - Predict next file user will open:
         * If editing main.py, likely to open utils.py next (import)
         * If viewing test file, likely to open implementation file
       - Pre-fetch context for predicted files
       - Cache hit rate: 60-70% for next file

    Flow:
    1. User opens file.py
    2. IDE sends file_opened event
    3. Background: Parse AST, extract context (500ms)
    4. User starts typing (3 seconds later)
    5. First completion request → Context already cached → Fast response (100ms)
    6. User switches to utils.py (predicted file)
    7. Context pre-fetched → Instant response (50ms)
    ```

    ---

    ## Key Talking Points

    ### What Makes This System Unique?

    1. **Latency-critical:** <300ms for good UX (vs chat at <500ms)
    2. **Context extraction:** Intelligent selection of relevant code context
    3. **Multi-tier caching:** Completion, prefix, KV cache for maximum efficiency
    4. **Security:** Real-time credential scanning (can't leak secrets)
    5. **Telemetry-driven:** Acceptance rate feedback loop to improve model

    ### What Would You Do Differently?

    - **Smaller context window:** Start with 4K tokens, expand to 8K if quality improves
    - **Edge deployment:** Deploy smaller models on-device for ultra-low latency (<50ms)
    - **Personalization:** Train per-user adapters (LoRA) for style matching
    - **Multi-modal:** Support images (screenshots), not just text
    - **Collaborative:** Learn from team patterns, not just individual

    ### Real-World References

    | System | Model Size | Context Window | Latency | Acceptance Rate |
    |--------|------------|----------------|---------|-----------------|
    | **GitHub Copilot** | 12B params | 8K tokens | 150-300ms | 40-50% |
    | **Cursor** | 7B-34B params | 32K tokens | 200-400ms | 45-55% |
    | **Replit Ghostwriter** | 3B params | 4K tokens | 100-200ms | 35-45% |
    | **Amazon CodeWhisperer** | Unknown (likely 7B) | 8K tokens | 200-300ms | 40-50% |
    | **Tabnine** | 1B-7B params | 4K tokens | 100-200ms | 30-40% |

=== "✅ Step 8: Summary"

    ## System Overview

    **AI Code Assistant** is a low-latency, context-aware code completion system that:

    1. **Integrates with IDEs** via LSP for precise context extraction
    2. **Provides instant suggestions** (<300ms) using small, optimized models (1B params)
    3. **Leverages multi-tier caching** (completion, prefix, KV cache) for 85% effective hit rate
    4. **Scans for credentials** using regex + ML to prevent leaked secrets
    5. **Scales to 10B completions/day** with 1,150 GPUs and auto-scaling
    6. **Learns from telemetry** (acceptance rate) to continuously improve

    ---

    ## Key Design Decisions

    | Decision | Rationale | Trade-off |
    |----------|-----------|-----------|
    | Small completion model (1B) | 100-200ms latency vs 500ms+ for 7B model | 10-20% lower quality, but acceptable for inline |
    | Fill-in-the-Middle (FIM) | Complete within code blocks, not just suffix | Requires FIM-trained model |
    | Prompt prefix caching | 70% hit rate → 3-5x speedup | Cache infrastructure complexity |
    | Speculative decoding | 2-3x faster inference | 5-10% quality loss |
    | LSP integration | Precise context (imports, symbols) | Requires IDE support |
    | Multi-tier caching | 85% effective hit rate | More complex cache invalidation |
    | Client-side debouncing | Reduce requests by 50-70% | Slight delay before triggering |

    ---

    ## Capacity Summary

    ```
    Scale:
    - 50M daily active developers
    - 10B code completions per day (115K req/sec)
    - 500M chat messages per day (5.8K req/sec)
    - Peak: 230K completion req/sec

    Infrastructure:
    - 1,150 A100 GPUs (completion: 280, chat: 870)
    - 300 TB storage (logs, models, cache)
    - 1.7 TB Redis cache (completions, sessions, AST)
    - 100 API gateway nodes
    - 50 context extraction workers

    Cost:
    - GPU: $2.07M/month
    - Storage: $30K/month
    - Networking: $50K/month
    - Total: ~$2.2M/month

    Performance:
    - P95 latency: 180ms (inline), 400ms (chat)
    - Cache hit rate: 85% effective
    - GPU utilization: 88%
    - Acceptance rate: 30-50%
    ```

    ---

    ## Next Steps (Post-Interview)

    If you had more time, discuss:

    1. **On-device inference:** Deploy small models on developer laptops for <50ms latency
    2. **Personalization:** Train per-user LoRA adapters to match coding style
    3. **Multi-modal support:** Accept screenshots, diagrams as input
    4. **Codebase indexing:** Vector database for semantic code search
    5. **Collaborative filtering:** Learn from team patterns, not just individual
    6. **Fine-tuning pipeline:** Continuously fine-tune on accepted completions
    7. **IDE-specific optimizations:** Different strategies for VSCode vs JetBrains
    8. **Privacy-preserving:** Federated learning to train without sending code to cloud

    ---

    ## Additional Resources

    **Papers:**
    - [Copilot: GitHub's Code Completion Model](https://arxiv.org/abs/2107.03374)
    - [CodeGen: Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474)
    - [InCoder: Generative Models for Code Infilling](https://arxiv.org/abs/2204.05999)
    - [Speculative Decoding: Faster LLM Inference](https://arxiv.org/abs/2211.17192)

    **Systems:**
    - GitHub Copilot: https://github.com/features/copilot
    - Cursor: https://cursor.sh
    - Replit Ghostwriter: https://replit.com/site/ghostwriter
    - Amazon CodeWhisperer: https://aws.amazon.com/codewhisperer

    **Open Source:**
    - vLLM (fast inference): https://github.com/vllm-project/vllm
    - Continue.dev (open-source Copilot): https://github.com/continuedev/continue
    - Tabby (self-hosted): https://github.com/TabbyML/tabby
