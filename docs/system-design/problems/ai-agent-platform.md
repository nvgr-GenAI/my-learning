# Design AI Agent Platform (like LangChain, AutoGPT, CrewAI)

An autonomous AI agent platform that enables agents to reason, plan, execute tools, maintain memory, and communicate with each other to solve complex tasks. Agents can break down goals, call external APIs/tools, learn from interactions, and coordinate multi-agent workflows.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K concurrent agents, 10M tool calls/day, 100M memory operations/day, 1M agent executions/day |
| **Key Challenges** | Agent reasoning (ReAct, Chain-of-Thought), tool discovery/execution, memory management, multi-agent orchestration, sandboxing |
| **Core Concepts** | ReAct prompting, function calling, vector memory, episodic memory, agent orchestration, tool registry, execution safety |
| **Companies** | LangChain, AutoGPT, BabyAGI, SuperAGI, CrewAI, Microsoft (Semantic Kernel), Anthropic (Claude Code) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Agent Creation** | Define agent with instructions, tools, memory | P0 (Must have) |
    | **Tool Calling** | Execute external functions/APIs from agent | P0 (Must have) |
    | **ReAct Loop** | Reasoning + Acting pattern for task solving | P0 (Must have) |
    | **Short-term Memory** | Context window management for conversations | P0 (Must have) |
    | **Long-term Memory** | Vector + graph memory for persistent knowledge | P0 (Must have) |
    | **Tool Discovery** | Search and register tools dynamically | P0 (Must have) |
    | **Execution Tracking** | Log agent steps, decisions, tool calls | P0 (Must have) |
    | **Error Handling** | Retry logic, fallbacks, graceful degradation | P0 (Must have) |
    | **Multi-Agent Orchestration** | Coordinate multiple agents on a task | P1 (Should have) |
    | **Agent Communication** | Message passing between agents | P1 (Should have) |
    | **Sandboxing** | Secure tool execution in isolated environment | P1 (Should have) |
    | **Agent Evaluation** | Metrics, benchmarks, testing framework | P1 (Should have) |
    | **Planning Strategies** | Chain-of-Thought, Tree-of-Thought | P2 (Nice to have) |
    | **Human-in-the-Loop** | Request human approval for sensitive actions | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training/fine-tuning infrastructure
    - Agent learning/reinforcement learning
    - Custom tool development UI
    - Visual agent builder (drag-and-drop)
    - Voice/video agent interactions

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Tool Execution)** | < 2s p95 for tool calls | Agents need fast feedback loops |
    | **Latency (Agent Response)** | < 10s p95 for single reasoning step | Balance speed with thorough reasoning |
    | **Reliability** | 99.5% success rate for tool execution | Failures cascade through agent loop |
    | **Availability** | 99.9% uptime | Production agents must be reliable |
    | **Scalability** | Support 100K agents, 100M tool calls/day | Enterprise scale requirements |
    | **Security** | Sandboxed execution, no data leakage | Tools can execute arbitrary code |
    | **Cost** | < $0.05 per agent execution (infra cost) | LLM calls are expensive |
    | **Observability** | Full execution traces for debugging | Agent debugging is hard |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Agent executions:
    - Daily agent runs: 1M executions/day
    - Execution QPS: 1M / 86,400 = ~12 exec/sec average
    - Peak QPS: 3x average = ~36 exec/sec

    Tool calls:
    - Average tools per agent execution: 10 tools
    - Daily tool calls: 1M × 10 = 10M tool calls/day
    - Tool call QPS: 10M / 86,400 = ~116 calls/sec
    - Peak: ~350 calls/sec

    LLM requests:
    - Average LLM calls per execution: 15 calls (ReAct loop iterations)
    - Daily LLM requests: 1M × 15 = 15M requests/day
    - LLM QPS: 15M / 86,400 = ~174 req/sec
    - Peak: ~520 req/sec

    Memory operations:
    - Memory writes per execution: 20 operations
    - Memory reads per execution: 50 operations
    - Daily writes: 1M × 20 = 20M writes/day
    - Daily reads: 1M × 50 = 50M reads/day
    - Read QPS: 50M / 86,400 = ~580 reads/sec
    - Write QPS: 20M / 86,400 = ~230 writes/sec

    Multi-agent coordination:
    - 20% of executions use multiple agents
    - Average agents per multi-agent task: 3 agents
    - Agent messages: 1M × 0.2 × 3 × 5 = 3M messages/day
    - Message QPS: 3M / 86,400 = ~35 msg/sec

    Concurrent executions:
    - Average execution time: 2 minutes
    - Concurrent agents: 12 exec/sec × 120 sec = ~1,440 concurrent
    - Peak concurrent: ~4,320 concurrent agents
    ```

    ### Storage Estimates

    ```
    Agent definitions:
    - Total agents: 100K agents
    - Agent metadata: 10 KB (instructions, tools, config)
    - 100K × 10 KB = 1 GB

    Tool registry:
    - Total tools: 10K tools
    - Tool definition: 5 KB (schema, description, code)
    - 10K × 5 KB = 50 MB

    Execution logs:
    - Daily executions: 1M
    - Average execution log: 50 KB (steps, decisions, tool calls)
    - Daily: 1M × 50 KB = 50 GB/day
    - 90 days retention: 50 GB × 90 = 4.5 TB

    Conversation history (short-term memory):
    - Active conversations: 100K
    - Average conversation: 20 messages × 2 KB = 40 KB
    - 100K × 40 KB = 4 GB
    - 30 days: 4 GB × 30 = 120 GB

    Long-term memory (vector):
    - Total embeddings: 100M embeddings
    - Embedding size: 1536 dim × 4 bytes = 6 KB
    - Metadata: 2 KB
    - 100M × 8 KB = 800 GB
    - With HNSW index overhead: ~1.2 TB

    Long-term memory (graph):
    - Nodes: 50M entities
    - Edges: 200M relationships
    - Node data: 50M × 1 KB = 50 GB
    - Edge data: 200M × 500 bytes = 100 GB
    - Total: 150 GB

    Tool execution artifacts:
    - Tool results cached for 24h
    - Average result: 10 KB
    - Cache 10M results = 100 GB

    Total: 1 GB (agents) + 50 MB (tools) + 4.5 TB (logs) + 120 GB (conversations) + 1.2 TB (vector) + 150 GB (graph) + 100 GB (cache) ≈ 6 TB
    ```

    ### Compute Estimates

    ```
    LLM inference:
    - 520 req/sec peak × 3,000 tokens/request = 1.56M tokens/sec
    - Using hosted API (OpenAI/Anthropic): ~$0.01 per 1K tokens
    - Daily cost: 15M requests × 3K tokens × $0.01 / 1000 = $450/day
    - Monthly: $13,500

    Tool execution (sandboxed containers):
    - 350 tool calls/sec peak
    - Average tool execution: 500ms
    - Concurrent containers: 350 × 0.5 = 175 containers
    - Container size: 2 CPU, 4 GB RAM
    - Cost: 175 × 2 CPU × $0.04/hour = $14/hour = $336/day = $10K/month

    Memory operations (vector search):
    - 580 searches/sec peak
    - Vector DB cluster: 10 nodes × 16 CPU, 64 GB RAM
    - Cost: 10 × 64 GB × $0.01/GB/hour = $6.40/hour = $154/day = $4.6K/month

    API servers:
    - 36 exec/sec + 350 tool calls/sec + 520 LLM req/sec = 906 req/sec
    - 20 API servers × 8 CPU, 16 GB RAM
    - Cost: 20 × $0.50/hour = $10/hour = $240/day = $7.2K/month

    Total monthly compute: $13.5K (LLM) + $10K (containers) + $4.6K (vector) + $7.2K (API) = $35.3K/month
    ```

    ### Bandwidth Estimates

    ```
    Agent requests:
    - 12 exec/sec × 10 KB (instructions + context) = 120 KB/sec ≈ 960 Kbps

    Tool calls:
    - 116 calls/sec × 5 KB (average tool request) = 580 KB/sec ≈ 4.6 Mbps
    - Tool responses: 116 × 10 KB = 1.16 MB/sec ≈ 9.3 Mbps

    LLM API calls:
    - 174 req/sec × 12 KB (prompt + response) = 2.1 MB/sec ≈ 16.8 Mbps

    Memory operations:
    - 580 reads/sec × 8 KB = 4.6 MB/sec ≈ 36.8 Mbps
    - 230 writes/sec × 8 KB = 1.8 MB/sec ≈ 14.4 Mbps

    Total ingress: ~22 Mbps
    Total egress: ~46 Mbps
    Internal: ~51 Mbps
    ```

    ---

    ## Key Assumptions

    1. Average agent execution takes 2 minutes and makes 10 tool calls
    2. 80% of agents use ReAct pattern, 20% use simpler patterns
    3. Tool execution takes 500ms average (range: 100ms - 5s)
    4. 70% of executions complete successfully, 20% retry, 10% fail
    5. Multi-agent tasks represent 20% of total executions
    6. Long-term memory hit rate: 30% (most agents use fresh context)
    7. Tool discovery reduces redundant tool creation by 40%

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separation of runtime and orchestration** - Agent execution independent from coordination
    2. **Pluggable tool system** - Dynamic tool registration and discovery
    3. **Multi-layer memory** - Short-term (context), long-term (vector + graph), episodic (execution history)
    4. **Sandboxed execution** - Isolate tool execution for security
    5. **Observable by default** - Full execution traces for debugging
    6. **Async by default** - Non-blocking tool execution and LLM calls

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            SDK[Agent SDK<br/>Python, TypeScript]
            CLI[CLI Tool]
            WebUI[Web Dashboard]
        end

        subgraph "API Gateway"
            LB[Load Balancer]
            Auth[Auth Service<br/>API keys, JWT]
            RateLimit[Rate Limiter]
        end

        subgraph "Agent Runtime"
            Agent_API[Agent API<br/>Create, execute, manage]
            Agent_Engine[Agent Engine<br/>ReAct loop execution]
            Planning[Planning Module<br/>CoT, ToT, decomposition]
            Reasoning[Reasoning Module<br/>Decision making]
        end

        subgraph "Tool System"
            Tool_Registry[Tool Registry<br/>Tool discovery, schemas]
            Tool_Executor[Tool Executor<br/>Sandboxed execution]
            Tool_Cache[Tool Cache<br/>Result caching]
        end

        subgraph "Memory System"
            Memory_Manager[Memory Manager<br/>Unified interface]
            ShortTerm[Short-term Memory<br/>Conversation context]
            LongTerm_Vector[Long-term Memory<br/>Vector DB<br/>Semantic search]
            LongTerm_Graph[Long-term Memory<br/>Graph DB<br/>Entity relationships]
            Episodic[Episodic Memory<br/>Execution history]
        end

        subgraph "LLM Gateway"
            LLM_Router[LLM Router<br/>Model selection, fallback]
            LLM_Cache[Prompt Cache<br/>Reuse common prompts]
            LLM_Pool[LLM Provider Pool<br/>OpenAI, Anthropic, local]
        end

        subgraph "Multi-Agent Orchestration"
            Orchestrator[Agent Orchestrator<br/>Multi-agent coordination]
            Message_Bus[Message Bus<br/>Agent communication]
            Task_Queue[Task Queue<br/>Work distribution]
        end

        subgraph "Execution Infrastructure"
            Sandbox[Sandbox Manager<br/>Docker, gVisor]
            Monitor[Execution Monitor<br/>Timeouts, errors]
            Retry[Retry Handler<br/>Backoff, circuit breaker]
        end

        subgraph "Storage"
            Agent_DB[(Agent Store<br/>PostgreSQL<br/>Agent definitions)]
            Tool_DB[(Tool Store<br/>PostgreSQL<br/>Tool schemas)]
            Execution_DB[(Execution Logs<br/>MongoDB<br/>Time-series)]
            Vector_DB[(Vector Store<br/>Pinecone/Weaviate<br/>Embeddings)]
            Graph_DB[(Graph Store<br/>Neo4j<br/>Entities)]
        end

        subgraph "Caching"
            Redis_State[Redis<br/>Agent state]
            Redis_Tool[Redis<br/>Tool results]
            Redis_Memory[Redis<br/>Hot memory]
        end

        subgraph "Observability"
            Trace[Tracing<br/>Jaeger<br/>Execution traces]
            Metrics[Metrics<br/>Prometheus<br/>Performance]
            Logs[Logs<br/>ELK Stack]
        end

        SDK --> LB
        CLI --> LB
        WebUI --> LB

        LB --> Auth
        Auth --> RateLimit
        RateLimit --> Agent_API

        Agent_API --> Agent_Engine
        Agent_API --> Orchestrator
        Agent_API --> Agent_DB

        Agent_Engine --> Planning
        Agent_Engine --> Reasoning
        Agent_Engine --> LLM_Router
        Agent_Engine --> Memory_Manager
        Agent_Engine --> Tool_Registry

        Planning --> LLM_Router
        Reasoning --> LLM_Router

        LLM_Router --> LLM_Cache
        LLM_Router --> LLM_Pool

        Tool_Registry --> Tool_DB
        Tool_Registry --> Tool_Executor
        Tool_Executor --> Tool_Cache
        Tool_Executor --> Sandbox
        Tool_Executor --> Monitor

        Tool_Cache --> Redis_Tool
        Monitor --> Retry

        Memory_Manager --> ShortTerm
        Memory_Manager --> LongTerm_Vector
        Memory_Manager --> LongTerm_Graph
        Memory_Manager --> Episodic
        Memory_Manager --> Redis_Memory

        ShortTerm --> Redis_State
        LongTerm_Vector --> Vector_DB
        LongTerm_Graph --> Graph_DB
        Episodic --> Execution_DB

        Orchestrator --> Message_Bus
        Orchestrator --> Task_Queue
        Message_Bus --> Agent_Engine

        Agent_Engine --> Trace
        Tool_Executor --> Trace
        Agent_Engine --> Metrics
        Agent_API --> Logs

        style LB fill:#e1f5ff
        style Redis_State fill:#fff4e1
        style Redis_Tool fill:#fff4e1
        style Redis_Memory fill:#fff4e1
        style Agent_DB fill:#ffe1e1
        style Tool_DB fill:#ffe1e1
        style Execution_DB fill:#ffe1e1
        style Vector_DB fill:#e8f5e9
        style Graph_DB fill:#e8f5e9
        style Sandbox fill:#f3e5f5
        style LLM_Pool fill:#e1f5ff
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Agent Engine (ReAct)** | Enables reasoning + acting loop, self-correction, tool use | Simple prompt chain (no reasoning), MRKL (less flexible) |
    | **Tool Registry** | Centralized tool discovery, avoid duplication, versioning | Hard-coded tools (not scalable), function libraries (no discovery) |
    | **Vector DB (Memory)** | Semantic search over past experiences, efficient retrieval | Keyword search (less relevant), no memory (poor context) |
    | **Graph DB (Memory)** | Entity relationships, knowledge graph, complex queries | SQL (poor for graphs), document DB (no relationships) |
    | **Sandbox (Docker/gVisor)** | Security isolation for arbitrary tool execution | No sandbox (unsafe), VM (too slow), WebAssembly (limited capabilities) |
    | **Message Bus (Multi-agent)** | Async communication, pub/sub, event-driven | Direct calls (tight coupling), database polling (latency) |
    | **MongoDB (Executions)** | Time-series logs, flexible schema, fast writes | PostgreSQL (less flexible), Elasticsearch (overkill for storage) |

    **Key Trade-off:** We chose **sandboxed execution** with Docker containers over native execution. This adds latency (500ms vs 10ms) but prevents malicious tool code from compromising the system. For production agent platforms, security must come first.

    ---

    ## API Design

    ### 1. Create Agent

    **Request:**
    ```http
    POST /v1/agents
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "name": "research-agent",
      "instructions": "You are a research agent. Search the web, analyze information, and provide comprehensive summaries.",
      "model": "gpt-4",
      "tools": [
        "web_search",
        "web_scraper",
        "summarizer"
      ],
      "memory": {
        "short_term": {
          "max_messages": 20
        },
        "long_term": {
          "enabled": true,
          "vector_store": "pinecone",
          "graph_store": "neo4j"
        }
      },
      "config": {
        "max_iterations": 15,
        "timeout": 300,
        "temperature": 0.7,
        "reasoning_strategy": "react"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "id": "agent_abc123",
      "name": "research-agent",
      "status": "active",
      "created_at": "2026-02-05T10:30:00Z",
      "tools_loaded": 3,
      "memory_initialized": true
    }
    ```

    ---

    ### 2. Execute Agent

    **Request:**
    ```http
    POST /v1/agents/{agent_id}/execute
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "input": "Research the latest developments in quantum computing and summarize the top 3 breakthroughs.",
      "context": {
        "user_id": "user_123",
        "session_id": "session_xyz"
      },
      "config": {
        "stream": true,
        "max_iterations": 10
      }
    }
    ```

    **Response (Streaming):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: text/event-stream
    Connection: keep-alive

    event: agent.start
    data: {"execution_id":"exec_def456","status":"running","timestamp":"2026-02-05T10:30:01Z"}

    event: agent.thought
    data: {"step":1,"type":"reasoning","content":"I need to search for recent quantum computing breakthroughs. Let me use the web_search tool."}

    event: agent.action
    data: {"step":2,"type":"tool_call","tool":"web_search","input":{"query":"quantum computing breakthroughs 2026","max_results":10}}

    event: tool.result
    data: {"step":3,"tool":"web_search","output":{"results":[{"title":"IBM achieves 1000-qubit processor","url":"..."},{"title":"Google demonstrates quantum error correction","url":"..."}]}}

    event: agent.thought
    data: {"step":4,"type":"reasoning","content":"Great, I found several recent articles. Let me scrape the top 3 to get more details."}

    event: agent.action
    data: {"step":5,"type":"tool_call","tool":"web_scraper","input":{"url":"https://example.com/ibm-1000-qubit"}}

    event: tool.result
    data: {"step":6,"tool":"web_scraper","output":{"content":"IBM announced a 1000-qubit quantum processor..."}}

    ...

    event: agent.answer
    data: {"step":15,"type":"final_answer","content":"Based on my research, here are the top 3 breakthroughs in quantum computing:\n\n1. **IBM 1000-qubit processor**: IBM achieved...\n2. **Google quantum error correction**: Google demonstrated...\n3. **China quantum network**: Chinese researchers..."}

    event: agent.complete
    data: {"execution_id":"exec_def456","status":"completed","iterations":8,"tool_calls":5,"tokens_used":3420,"duration_ms":45000}
    ```

    ---

    ### 3. Register Tool

    **Request:**
    ```http
    POST /v1/tools
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "name": "web_search",
      "description": "Search the web using a search engine API. Returns top results with titles, URLs, and snippets.",
      "category": "web",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query"
          },
          "max_results": {
            "type": "integer",
            "default": 10
          }
        },
        "required": ["query"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "snippet": {"type": "string"}
              }
            }
          }
        }
      },
      "executor": {
        "type": "http",
        "config": {
          "url": "https://api.search.com/search",
          "method": "GET",
          "headers": {
            "Authorization": "Bearer ${SEARCH_API_KEY}"
          },
          "timeout": 5000
        }
      },
      "sandbox": {
        "enabled": true,
        "network_access": true,
        "max_memory_mb": 512,
        "max_cpu_ms": 5000
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "id": "tool_xyz789",
      "name": "web_search",
      "status": "active",
      "version": "1.0.0",
      "created_at": "2026-02-05T10:30:00Z"
    }
    ```

    ---

    ### 4. Query Memory

    **Request:**
    ```http
    POST /v1/agents/{agent_id}/memory/query
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "query": "What did we discuss about quantum computing?",
      "memory_types": ["long_term", "episodic"],
      "limit": 10
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "results": [
        {
          "type": "long_term",
          "source": "vector",
          "content": "Quantum computing uses qubits that can exist in superposition...",
          "relevance_score": 0.92,
          "timestamp": "2026-02-01T15:30:00Z"
        },
        {
          "type": "episodic",
          "source": "execution",
          "content": "In execution exec_abc123, researched quantum breakthroughs including IBM 1000-qubit processor",
          "relevance_score": 0.88,
          "timestamp": "2026-02-05T10:30:00Z"
        }
      ],
      "total": 2
    }
    ```

    ---

    ### 5. Create Multi-Agent Task

    **Request:**
    ```http
    POST /v1/orchestration/tasks
    Content-Type: application/json
    Authorization: Bearer <api_key>

    {
      "name": "research-and-report",
      "description": "Research a topic and create a comprehensive report",
      "agents": [
        {
          "agent_id": "agent_researcher",
          "role": "researcher",
          "task": "Research quantum computing breakthroughs"
        },
        {
          "agent_id": "agent_analyzer",
          "role": "analyzer",
          "task": "Analyze technical details and implications",
          "depends_on": ["researcher"]
        },
        {
          "agent_id": "agent_writer",
          "role": "writer",
          "task": "Write a comprehensive report",
          "depends_on": ["analyzer"]
        }
      ],
      "coordination": {
        "type": "sequential",
        "share_memory": true
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "task_id": "task_multi_123",
      "status": "running",
      "agents": 3,
      "started_at": "2026-02-05T10:30:00Z"
    }
    ```

    ---

    ## Database Schema

    ### Agent Store (PostgreSQL)

    ```sql
    CREATE TABLE agents (
        id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        user_id VARCHAR(50) NOT NULL,
        instructions TEXT,
        model VARCHAR(100) DEFAULT 'gpt-4',
        reasoning_strategy VARCHAR(50) DEFAULT 'react',
        config JSONB,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_id (user_id),
        INDEX idx_status (status)
    );

    CREATE TABLE agent_tools (
        id SERIAL PRIMARY KEY,
        agent_id VARCHAR(50) REFERENCES agents(id) ON DELETE CASCADE,
        tool_id VARCHAR(50) NOT NULL,
        enabled BOOLEAN DEFAULT true,
        config JSONB,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_agent_id (agent_id),
        INDEX idx_tool_id (tool_id)
    );

    CREATE TABLE agent_memory_config (
        id SERIAL PRIMARY KEY,
        agent_id VARCHAR(50) REFERENCES agents(id) ON DELETE CASCADE,
        memory_type VARCHAR(50) NOT NULL, -- short_term, long_term_vector, long_term_graph, episodic
        enabled BOOLEAN DEFAULT true,
        config JSONB,
        INDEX idx_agent_id (agent_id)
    );
    ```

    ---

    ### Tool Registry (PostgreSQL)

    ```sql
    CREATE TABLE tools (
        id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        description TEXT,
        category VARCHAR(100),
        version VARCHAR(20) DEFAULT '1.0.0',
        user_id VARCHAR(50), -- NULL for system tools
        input_schema JSONB NOT NULL,
        output_schema JSONB NOT NULL,
        executor_config JSONB NOT NULL,
        sandbox_config JSONB NOT NULL,
        status VARCHAR(20) DEFAULT 'active',
        usage_count BIGINT DEFAULT 0,
        avg_execution_time_ms INT,
        success_rate FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_name (name),
        INDEX idx_category (category),
        INDEX idx_user_id (user_id)
    );

    CREATE TABLE tool_executions (
        id VARCHAR(50) PRIMARY KEY,
        tool_id VARCHAR(50) REFERENCES tools(id),
        agent_id VARCHAR(50),
        execution_id VARCHAR(50),
        input JSONB,
        output JSONB,
        status VARCHAR(20), -- success, failure, timeout
        error TEXT,
        execution_time_ms INT,
        sandbox_id VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_tool_id (tool_id),
        INDEX idx_execution_id (execution_id),
        INDEX idx_created_at (created_at)
    );
    ```

    ---

    ### Execution Logs (MongoDB)

    ```javascript
    // executions collection
    {
      _id: "exec_def456",
      agent_id: "agent_abc123",
      user_id: "user_123",
      input: "Research quantum computing breakthroughs",
      status: "completed", // running, completed, failed

      // Execution trace
      steps: [
        {
          step_number: 1,
          type: "thought",
          content: "I need to search for recent quantum computing breakthroughs",
          timestamp: ISODate("2026-02-05T10:30:01Z")
        },
        {
          step_number: 2,
          type: "action",
          tool: "web_search",
          tool_call_id: "tc_123",
          input: { query: "quantum computing breakthroughs 2026" },
          timestamp: ISODate("2026-02-05T10:30:02Z")
        },
        {
          step_number: 3,
          type: "observation",
          tool_call_id: "tc_123",
          output: { results: [...] },
          execution_time_ms: 450,
          timestamp: ISODate("2026-02-05T10:30:03Z")
        }
      ],

      // Summary
      summary: {
        iterations: 8,
        tool_calls: 5,
        tokens_used: 3420,
        duration_ms: 45000,
        final_answer: "Based on my research..."
      },

      // Metadata
      config: {
        max_iterations: 15,
        timeout: 300,
        temperature: 0.7
      },

      created_at: ISODate("2026-02-05T10:30:00Z"),
      completed_at: ISODate("2026-02-05T10:30:45Z")
    }

    // Index on agent_id, user_id, created_at for queries
    db.executions.createIndex({ agent_id: 1, created_at: -1 })
    db.executions.createIndex({ user_id: 1, created_at: -1 })
    db.executions.createIndex({ status: 1, created_at: -1 })
    ```

    ---

    ### Conversation History (Short-term Memory)

    ```sql
    -- In Redis for hot data, PostgreSQL for persistence
    CREATE TABLE conversations (
        id VARCHAR(50) PRIMARY KEY,
        agent_id VARCHAR(50) REFERENCES agents(id),
        user_id VARCHAR(50) NOT NULL,
        session_id VARCHAR(50),
        title TEXT,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_message_at TIMESTAMP,
        INDEX idx_agent_user (agent_id, user_id),
        INDEX idx_session (session_id)
    );

    CREATE TABLE messages (
        id VARCHAR(50) PRIMARY KEY,
        conversation_id VARCHAR(50) REFERENCES conversations(id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL, -- user, assistant, system, tool
        content TEXT,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_conversation_id (conversation_id),
        INDEX idx_created_at (created_at)
    );
    ```

    ---

    ### Long-term Memory (Vector Store)

    ```python
    # Pinecone/Weaviate schema
    {
      "id": "mem_vec_123",
      "vector": [0.123, 0.456, ...],  # 1536 dimensions
      "metadata": {
        "agent_id": "agent_abc123",
        "user_id": "user_123",
        "content": "Quantum computing uses qubits...",
        "source": "conversation",
        "timestamp": "2026-02-05T10:30:00Z",
        "relevance_score": 0.92,
        "tags": ["quantum", "computing", "qubits"]
      }
    }
    ```

    ---

    ### Long-term Memory (Graph Store - Neo4j)

    ```cypher
    // Entity nodes
    CREATE (e:Entity {
      id: 'entity_qc_001',
      type: 'concept',
      name: 'Quantum Computing',
      description: 'Computing paradigm using quantum mechanics',
      agent_id: 'agent_abc123',
      created_at: datetime()
    })

    // Relationship edges
    CREATE (e1:Entity {name: 'Quantum Computing'})
    CREATE (e2:Entity {name: 'Qubits'})
    CREATE (e1)-[:USES {
      confidence: 0.95,
      source: 'execution_exec_def456',
      created_at: datetime()
    }]->(e2)

    // Indexes
    CREATE INDEX entity_id FOR (e:Entity) ON (e.id)
    CREATE INDEX entity_agent FOR (e:Entity) ON (e.agent_id)
    CREATE INDEX entity_type FOR (e:Entity) ON (e.type)
    ```

    ---

    ## Data Flow Diagrams

    ### Agent Execution Flow (ReAct Loop)

    ```mermaid
    sequenceDiagram
        participant Client
        participant API as Agent API
        participant Engine as Agent Engine
        participant LLM as LLM Gateway
        participant Tools as Tool Executor
        participant Memory as Memory Manager
        participant DB as Execution DB

        Client->>API: POST /agents/{id}/execute
        API->>Engine: Initialize execution
        Engine->>Memory: Load context + memory
        Memory-->>Engine: Context data

        loop ReAct Loop (max 15 iterations)
            Engine->>LLM: Generate thought (reasoning)
            LLM-->>Engine: "I need to search..."
            Engine->>DB: Log thought step

            Engine->>LLM: Decide action (tool call)
            LLM-->>Engine: call web_search(query="...")
            Engine->>DB: Log action step

            Engine->>Tools: Execute tool
            Tools->>Tools: Spin up sandbox
            Tools->>Tools: Run tool code
            Tools-->>Engine: Tool result
            Engine->>DB: Log observation step

            Engine->>Memory: Store observation

            Engine->>LLM: Should continue?

            alt Has final answer
                LLM-->>Engine: Final answer ready
            else Need more actions
                LLM-->>Engine: Continue with next thought
            end
        end

        Engine->>Memory: Store execution in episodic memory
        Engine->>DB: Mark execution complete
        Engine-->>API: Execution result
        API-->>Client: Stream final response
    ```

    **Flow Explanation:**

    1. **Initialize execution** - Load agent config, tools, memory
    2. **ReAct loop** - Iterative reasoning + acting cycle:
       - **Thought**: LLM reasons about what to do next
       - **Action**: LLM decides which tool to call with what input
       - **Observation**: Tool executes and returns result
       - **Repeat** until LLM generates final answer or max iterations
    3. **Store memory** - Save execution trace to episodic memory
    4. **Return result** - Stream final answer to client

    **Latency:** ~30-60 seconds for complete execution (8 iterations × 5-7s per iteration)

    ---

    ### Tool Discovery and Execution

    ```mermaid
    sequenceDiagram
        participant Agent as Agent Engine
        participant Registry as Tool Registry
        participant Cache as Tool Cache
        participant Sandbox as Sandbox Manager
        participant Monitor as Execution Monitor

        Agent->>Registry: Search tools by description
        Registry->>Registry: Semantic search (embedding similarity)
        Registry-->>Agent: Matching tools [web_search, web_scraper]

        Agent->>Agent: Select tool: web_search
        Agent->>Cache: Check cached result

        alt Cache HIT
            Cache-->>Agent: Return cached result (< 10ms)
        else Cache MISS
            Cache-->>Agent: null

            Agent->>Sandbox: Request execution environment
            Sandbox->>Sandbox: Spin up Docker container
            Sandbox-->>Agent: Container ready (sandbox_abc123)

            Agent->>Monitor: Start execution timer
            Agent->>Sandbox: Execute tool with input

            par Parallel execution
                Sandbox->>Sandbox: Run tool code
                Monitor->>Monitor: Check timeout (5s max)
            end

            alt Success
                Sandbox-->>Agent: Tool result
                Agent->>Cache: Store result (TTL: 24h)
            else Timeout
                Monitor->>Sandbox: Kill container
                Monitor-->>Agent: Timeout error
                Agent->>Agent: Retry with exponential backoff
            else Error
                Sandbox-->>Agent: Execution error
                Agent->>Agent: Log error, try alternative tool
            end

            Agent->>Sandbox: Cleanup container
        end

        Agent-->>Agent: Continue with result
    ```

    **Flow Explanation:**

    1. **Tool discovery** - Search registry by semantic similarity
    2. **Cache check** - Avoid redundant execution for same input
    3. **Sandbox spin-up** - Isolated Docker container (500ms overhead)
    4. **Execution** - Run tool code with timeout monitoring
    5. **Error handling** - Retry logic, fallback to alternative tools
    6. **Cleanup** - Remove container after execution

    **Latency:**
    - Cache hit: < 10ms
    - Cache miss: 500ms (sandbox) + tool execution time (avg 500ms) = ~1s

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section covers critical components that enable autonomous agent behavior. Each requires deep understanding of prompting strategies, memory architectures, and distributed execution.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **ReAct Prompting** | How do agents reason and act? | ReAct loop: Thought → Action → Observation pattern |
    | **Tool Calling** | How to execute external functions? | Function calling API with JSON schema + sandboxing |
    | **Memory Architecture** | How do agents remember? | Multi-layer: Short-term (context), Long-term (vector + graph), Episodic (history) |
    | **Multi-Agent Orchestration** | How do agents collaborate? | Message bus + task queue with dependency management |

    ---

    === "🧠 ReAct Prompting & Chain-of-Thought"

        ## The Challenge

        Agents need to **reason about complex tasks** and **break them down into actionable steps**. Simple prompting fails for multi-step problems requiring tool use, planning, and self-correction.

        **The Problem:**

        - "One-shot" prompts can't handle complex tasks
        - No way to use external tools/APIs
        - Can't self-correct when wrong
        - Poor at planning multi-step solutions

        **The Solution: ReAct (Reasoning + Acting)**

        ReAct interleaves reasoning (thinking about what to do) with acting (executing tools) in a loop until the task is solved.

        ---

        ## ReAct Pattern

        **Core concept:** Agent alternates between three states:

        1. **Thought** - Reason about current situation, plan next action
        2. **Action** - Execute a tool with specific input
        3. **Observation** - Receive tool result, incorporate into reasoning

        **Visualization:**

        ```
        User Input: "What's the weather in Paris and should I bring an umbrella?"

        Thought 1: I need to find the current weather in Paris. I'll use the weather tool.
        Action 1: get_weather(location="Paris, France")
        Observation 1: {"temp": 12, "condition": "rainy", "precipitation": 80%}

        Thought 2: It's rainy with 80% precipitation. I should recommend an umbrella.
        Action 2: ANSWER

        Final Answer: The weather in Paris is rainy (12°C) with 80% chance of precipitation. Yes, you should definitely bring an umbrella!
        ```

        ---

        ## Implementation: ReAct Prompt Template

        ```python
        REACT_SYSTEM_PROMPT = """You are an AI agent that can reason and act to solve tasks.

        You have access to the following tools:
        {tool_descriptions}

        Use the following format:

        Thought: [Your reasoning about what to do next]
        Action: [The tool to use, formatted as tool_name(arg1="value1", arg2="value2")]
        Observation: [The result returned by the tool]
        ... (repeat Thought/Action/Observation as needed)
        Thought: [Final reasoning]
        Final Answer: [Your answer to the user]

        Important:
        - Always start with a Thought before taking an Action
        - Use ONLY the tools provided - do not make up tool names
        - If a tool fails, try a different approach
        - Keep thoughts concise but clear
        - When you have enough information, provide a Final Answer

        Begin!

        Question: {user_input}
        """

        class ReactAgent:
            """ReAct agent implementation"""

            def __init__(self, tools, llm, max_iterations=15):
                self.tools = {tool.name: tool for tool in tools}
                self.llm = llm
                self.max_iterations = max_iterations

            def execute(self, user_input: str) -> dict:
                """Execute ReAct loop"""

                # Build tool descriptions
                tool_descriptions = "\n".join([
                    f"- {name}: {tool.description}"
                    for name, tool in self.tools.items()
                ])

                # Initialize conversation
                prompt = REACT_SYSTEM_PROMPT.format(
                    tool_descriptions=tool_descriptions,
                    user_input=user_input
                )

                conversation_history = [{"role": "system", "content": prompt}]
                steps = []

                for iteration in range(self.max_iterations):
                    # Generate next thought/action
                    response = self.llm.complete(conversation_history)
                    text = response.content

                    # Parse response
                    if "Final Answer:" in text:
                        # Extract final answer
                        final_answer = text.split("Final Answer:")[1].strip()
                        return {
                            "status": "completed",
                            "answer": final_answer,
                            "steps": steps,
                            "iterations": iteration + 1
                        }

                    # Parse thought
                    if "Thought:" in text:
                        thought = text.split("Thought:")[1].split("Action:")[0].strip()
                        steps.append({
                            "type": "thought",
                            "content": thought
                        })

                    # Parse and execute action
                    if "Action:" in text:
                        action_text = text.split("Action:")[1].split("\n")[0].strip()

                        try:
                            # Parse tool call: tool_name(arg1="value1", arg2="value2")
                            tool_name, tool_input = self._parse_action(action_text)

                            steps.append({
                                "type": "action",
                                "tool": tool_name,
                                "input": tool_input
                            })

                            # Execute tool
                            tool = self.tools[tool_name]
                            result = tool.execute(**tool_input)

                            steps.append({
                                "type": "observation",
                                "output": result
                            })

                            # Add observation to conversation
                            observation_text = f"\nObservation: {result}\n"
                            conversation_history.append({
                                "role": "assistant",
                                "content": text + observation_text
                            })

                        except Exception as e:
                            # Tool execution failed
                            error_msg = f"Error: {str(e)}"
                            steps.append({
                                "type": "error",
                                "error": error_msg
                            })

                            conversation_history.append({
                                "role": "assistant",
                                "content": text + f"\nObservation: {error_msg}\n"
                            })

                # Max iterations reached
                return {
                    "status": "max_iterations",
                    "answer": "I couldn't complete the task within the iteration limit.",
                    "steps": steps,
                    "iterations": self.max_iterations
                }

            def _parse_action(self, action_text: str) -> tuple[str, dict]:
                """Parse action string into tool name and input dict"""
                # Example: get_weather(location="Paris, France")
                import re

                match = re.match(r'(\w+)\((.*)\)', action_text)
                if not match:
                    raise ValueError(f"Invalid action format: {action_text}")

                tool_name = match.group(1)
                args_str = match.group(2)

                # Parse arguments
                tool_input = {}
                for arg in args_str.split(','):
                    key, value = arg.split('=')
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    tool_input[key] = value

                return tool_name, tool_input
        ```

        ---

        ## Chain-of-Thought (CoT) Prompting

        **Concept:** Encourage model to "think step by step" before answering.

        **Simple CoT:**
        ```
        User: What is 15% of 80?

        Agent: Let me think step by step:
        1. 15% means 15/100 = 0.15
        2. 0.15 × 80 = 12

        Answer: 12
        ```

        **ReAct is CoT + Tool Use:**
        - CoT: Thinking steps
        - ReAct: Thinking + Acting (using tools)

        ---

        ## Advanced: Tree-of-Thought (ToT)

        **Concept:** Explore multiple reasoning paths simultaneously, backtrack if needed.

        ```python
        class TreeOfThoughtAgent:
            """Explore multiple reasoning paths"""

            def execute(self, problem: str, max_depth=3, branches=3):
                """
                Generate multiple thought paths, evaluate each, pick best

                Args:
                    problem: Problem to solve
                    max_depth: How many steps to explore
                    branches: How many alternatives per step
                """

                # Generate multiple initial thoughts
                thoughts = []
                for _ in range(branches):
                    thought = self.llm.complete(
                        f"Think of one approach to solve: {problem}"
                    )
                    thoughts.append({
                        "content": thought,
                        "score": self._evaluate_thought(thought, problem),
                        "depth": 1
                    })

                # Expand best thoughts
                for depth in range(2, max_depth + 1):
                    new_thoughts = []

                    # Expand top N thoughts
                    top_thoughts = sorted(thoughts, key=lambda x: x["score"], reverse=True)[:branches]

                    for thought in top_thoughts:
                        # Generate next steps
                        for _ in range(branches):
                            next_step = self.llm.complete(
                                f"Given this approach:\n{thought['content']}\n\nWhat's the next step?"
                            )
                            new_thoughts.append({
                                "content": thought["content"] + "\n" + next_step,
                                "score": self._evaluate_thought(next_step, problem),
                                "depth": depth
                            })

                    thoughts.extend(new_thoughts)

                # Return best path
                best_thought = max(thoughts, key=lambda x: x["score"])
                return best_thought["content"]

            def _evaluate_thought(self, thought: str, problem: str) -> float:
                """Evaluate how promising this thought is"""
                response = self.llm.complete(
                    f"Rate how promising this approach is for solving '{problem}':\n{thought}\n\nScore (0-1):"
                )
                return float(response.strip())
        ```

        **When to use:**
        - Complex problems with multiple valid approaches
        - High-stakes decisions requiring thorough exploration
        - Creative tasks (writing, design, brainstorming)

        **Trade-off:** Higher quality but 5-10x more LLM calls (expensive)

        ---

        ## Comparison: Prompting Strategies

        | Strategy | Complexity | Tool Use | Self-Correction | Cost | Best For |
        |----------|-----------|----------|----------------|------|----------|
        | **Zero-shot** | Low | ❌ | ❌ | 1x | Simple Q&A |
        | **Chain-of-Thought** | Medium | ❌ | ✅ | 1.5x | Math, logic problems |
        | **ReAct** | High | ✅ | ✅ | 3-5x | **Multi-step tasks (Recommended)** |
        | **Tree-of-Thought** | Very High | ✅ | ✅ | 10-20x | Complex planning, creative tasks |

    === "🔧 Tool Calling & Function Execution"

        ## The Challenge

        Agents need to **interact with external systems** (APIs, databases, code execution) to accomplish real-world tasks. This requires:

        - Structured tool definitions (schemas)
        - Safe execution (sandboxing)
        - Error handling (retries, fallbacks)
        - Result parsing and validation

        ---

        ## Tool Definition Format

        **OpenAI Function Calling Format:**

        ```python
        weather_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state/country, e.g. 'San Francisco, CA'"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }

        # LLM generates function call
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=[weather_tool],
            tool_choice="auto"
        )

        # Response includes tool call
        tool_call = response.choices[0].message.tool_calls[0]
        # tool_call.function.name = "get_weather"
        # tool_call.function.arguments = '{"location": "Paris, France", "units": "celsius"}'
        ```

        **Anthropic Tool Calling Format:**

        ```python
        tool_definition = {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }

        # LLM generates tool use
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=[tool_definition]
        )

        # Response includes tool use
        tool_use = response.content[0]
        # tool_use.type = "tool_use"
        # tool_use.name = "get_weather"
        # tool_use.input = {"location": "Paris, France", "units": "celsius"}
        ```

        ---

        ## Tool Execution Architecture

        ```python
        from enum import Enum
        from typing import Any, Dict
        import docker
        import json

        class ToolExecutor:
            """Sandboxed tool execution with Docker"""

            def __init__(self):
                self.docker_client = docker.from_env()
                self.execution_cache = {}  # Tool result caching

            def execute(
                self,
                tool_name: str,
                tool_input: Dict[str, Any],
                sandbox_config: Dict[str, Any]
            ) -> Dict[str, Any]:
                """
                Execute tool in sandboxed environment

                Args:
                    tool_name: Name of tool to execute
                    tool_input: Input parameters
                    sandbox_config: Sandbox settings (memory, CPU, network)

                Returns:
                    Tool execution result
                """

                # Check cache
                cache_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
                if cache_key in self.execution_cache:
                    return self.execution_cache[cache_key]

                try:
                    # Create sandbox container
                    container = self.docker_client.containers.run(
                        image="agent-tool-executor:latest",
                        command=[
                            "python", "-c",
                            f"from tools import {tool_name}; "
                            f"print({tool_name}(**{tool_input}))"
                        ],
                        mem_limit=f"{sandbox_config['max_memory_mb']}m",
                        cpu_period=100000,
                        cpu_quota=sandbox_config['max_cpu_ms'] * 100,
                        network_mode="bridge" if sandbox_config['network_access'] else "none",
                        detach=True,
                        remove=True
                    )

                    # Wait for completion with timeout
                    result = container.wait(timeout=sandbox_config.get('timeout', 30))

                    # Get output
                    logs = container.logs().decode('utf-8')

                    if result['StatusCode'] == 0:
                        # Success
                        output = json.loads(logs)

                        # Cache result
                        self.execution_cache[cache_key] = {
                            "status": "success",
                            "output": output
                        }

                        return {
                            "status": "success",
                            "output": output
                        }
                    else:
                        # Error
                        return {
                            "status": "error",
                            "error": logs
                        }

                except docker.errors.ContainerError as e:
                    return {
                        "status": "error",
                        "error": f"Container error: {str(e)}"
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Execution error: {str(e)}"
                    }
        ```

        ---

        ## Tool Protocol: HTTP vs Code

        **Option 1: HTTP Tools (Recommended for external APIs)**

        ```python
        {
          "name": "web_search",
          "executor": {
            "type": "http",
            "config": {
              "url": "https://api.search.com/search",
              "method": "POST",
              "headers": {
                "Authorization": "Bearer ${API_KEY}"
              },
              "body_template": {
                "query": "{query}",
                "max_results": "{max_results}"
              },
              "timeout": 5000
            }
          }
        }
        ```

        **Option 2: Code Tools (Python, JavaScript)**

        ```python
        {
          "name": "calculate",
          "executor": {
            "type": "code",
            "language": "python",
            "code": """
        def execute(expression: str) -> float:
            import ast
            import operator as op

            # Safe math expression evaluator
            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](
                        eval_expr(node.left),
                        eval_expr(node.right)
                    )
                else:
                    raise ValueError('Unsafe expression')

            return eval_expr(ast.parse(expression, mode='eval').body)
            """
          }
        }
        ```

        ---

        ## Error Handling & Retries

        ```python
        class ResilientToolExecutor:
            """Tool executor with retry logic and fallbacks"""

            def __init__(self, executor, max_retries=3):
                self.executor = executor
                self.max_retries = max_retries

            def execute_with_retry(self, tool_name: str, tool_input: dict) -> dict:
                """Execute with exponential backoff retry"""

                for attempt in range(self.max_retries):
                    try:
                        result = self.executor.execute(tool_name, tool_input)

                        if result["status"] == "success":
                            return result

                        # Transient error - retry
                        if self._is_retryable_error(result["error"]):
                            wait_time = 2 ** attempt  # Exponential backoff
                            time.sleep(wait_time)
                            continue

                        # Permanent error - don't retry
                        return result

                    except TimeoutError:
                        # Timeout - try with increased timeout
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {
                            "status": "error",
                            "error": "Tool execution timeout after retries"
                        }

                return {
                    "status": "error",
                    "error": "Max retries exceeded"
                }

            def _is_retryable_error(self, error: str) -> bool:
                """Determine if error is transient"""
                retryable_patterns = [
                    "timeout",
                    "connection refused",
                    "rate limit",
                    "503 service unavailable",
                    "429 too many requests"
                ]
                return any(pattern in error.lower() for pattern in retryable_patterns)
        ```

        ---

        ## Tool Discovery

        ```python
        class ToolRegistry:
            """Semantic tool discovery using embeddings"""

            def __init__(self, vector_db, embedding_model):
                self.vector_db = vector_db
                self.embedding_model = embedding_model

            def register_tool(self, tool: dict):
                """Register tool with semantic indexing"""

                # Create searchable description
                description = f"{tool['name']}: {tool['description']}"

                # Generate embedding
                embedding = self.embedding_model.embed(description)

                # Store in vector DB
                self.vector_db.upsert({
                    "id": tool["id"],
                    "vector": embedding,
                    "metadata": tool
                })

            def search_tools(self, query: str, limit: int = 5) -> list:
                """Search for relevant tools"""

                # Embed query
                query_embedding = self.embedding_model.embed(query)

                # Vector search
                results = self.vector_db.query(
                    vector=query_embedding,
                    top_k=limit,
                    include_metadata=True
                )

                return [
                    {
                        "tool": result.metadata,
                        "relevance_score": result.score
                    }
                    for result in results.matches
                ]

        # Usage
        registry = ToolRegistry(pinecone_client, openai_embeddings)

        # Register tool
        registry.register_tool({
            "id": "tool_web_search",
            "name": "web_search",
            "description": "Search the web using Google Search API. Returns top results with titles, URLs, and snippets.",
            "category": "web",
            "input_schema": {...}
        })

        # Discover tools
        tools = registry.search_tools("I need to find information online", limit=3)
        # Returns: [web_search, web_scraper, url_fetcher]
        ```

    === "🧠 Memory Architecture"

        ## The Challenge

        Agents need multiple types of memory:

        1. **Short-term** - Current conversation context (last 10-20 messages)
        2. **Long-term** - Persistent knowledge across conversations
        3. **Episodic** - Memory of past executions and their outcomes

        Each type has different storage requirements and access patterns.

        ---

        ## Memory Hierarchy

        ```
        ┌─────────────────────────────────────────────┐
        │         Short-Term Memory (Context)         │
        │   Redis (hot) + PostgreSQL (persistence)    │
        │   Last 20 messages, current execution       │
        │   Latency: < 10ms                          │
        └─────────────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────────────┐
        │      Long-Term Memory (Vector Store)        │
        │   Pinecone/Weaviate - Semantic search      │
        │   Past conversations, facts learned         │
        │   Latency: 50-100ms                        │
        └─────────────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────────────┐
        │      Long-Term Memory (Graph Store)         │
        │   Neo4j - Entity relationships             │
        │   Knowledge graph, entity connections       │
        │   Latency: 100-200ms                       │
        └─────────────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────────────┐
        │      Episodic Memory (Execution Logs)       │
        │   MongoDB - Time-series logs               │
        │   Past execution traces, tool calls         │
        │   Latency: 200-500ms                       │
        └─────────────────────────────────────────────┘
        ```

        ---

        ## Short-Term Memory (Conversation Context)

        ```python
        class ShortTermMemory:
            """Manage conversation context window"""

            def __init__(self, redis_client, max_messages=20):
                self.redis = redis_client
                self.max_messages = max_messages

            def add_message(self, conversation_id: str, message: dict):
                """Add message to conversation history"""

                key = f"conversation:{conversation_id}"

                # Add to list
                self.redis.lpush(key, json.dumps(message))

                # Trim to max length
                self.redis.ltrim(key, 0, self.max_messages - 1)

                # Set expiration (7 days)
                self.redis.expire(key, 7 * 24 * 3600)

            def get_context(self, conversation_id: str, limit: int = None) -> list:
                """Get recent messages"""

                key = f"conversation:{conversation_id}"

                # Get from Redis
                messages = self.redis.lrange(key, 0, (limit or self.max_messages) - 1)

                return [json.loads(msg) for msg in reversed(messages)]

            def summarize_old_context(self, conversation_id: str, llm):
                """Summarize old messages to save tokens"""

                messages = self.get_context(conversation_id, limit=50)

                if len(messages) > self.max_messages:
                    # Summarize old messages
                    old_messages = messages[:30]
                    summary = llm.complete(
                        f"Summarize this conversation history concisely:\n{old_messages}"
                    )

                    # Replace old messages with summary
                    new_context = [
                        {"role": "system", "content": f"Previous conversation summary: {summary}"}
                    ] + messages[30:]

                    # Update Redis
                    key = f"conversation:{conversation_id}"
                    self.redis.delete(key)
                    for msg in reversed(new_context):
                        self.redis.lpush(key, json.dumps(msg))
        ```

        ---

        ## Long-Term Memory (Vector Store)

        ```python
        class VectorMemory:
            """Semantic long-term memory using vector embeddings"""

            def __init__(self, vector_db, embedding_model):
                self.vector_db = vector_db
                self.embedding_model = embedding_model

            def store(
                self,
                agent_id: str,
                content: str,
                metadata: dict = None
            ) -> str:
                """Store memory with embedding"""

                # Generate embedding
                embedding = self.embedding_model.embed(content)

                # Generate ID
                memory_id = f"mem_{agent_id}_{uuid.uuid4()}"

                # Store in vector DB
                self.vector_db.upsert({
                    "id": memory_id,
                    "vector": embedding,
                    "metadata": {
                        "agent_id": agent_id,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat(),
                        **(metadata or {})
                    }
                })

                return memory_id

            def search(
                self,
                agent_id: str,
                query: str,
                limit: int = 10,
                filter_metadata: dict = None
            ) -> list:
                """Search for relevant memories"""

                # Generate query embedding
                query_embedding = self.embedding_model.embed(query)

                # Build filter
                filter_dict = {"agent_id": agent_id}
                if filter_metadata:
                    filter_dict.update(filter_metadata)

                # Vector search
                results = self.vector_db.query(
                    vector=query_embedding,
                    top_k=limit,
                    filter=filter_dict,
                    include_metadata=True
                )

                return [
                    {
                        "id": result.id,
                        "content": result.metadata["content"],
                        "relevance_score": result.score,
                        "timestamp": result.metadata["timestamp"]
                    }
                    for result in results.matches
                ]

            def forget(self, memory_ids: list):
                """Delete memories"""
                self.vector_db.delete(ids=memory_ids)
        ```

        ---

        ## Long-Term Memory (Graph Store)

        ```python
        class GraphMemory:
            """Entity relationship memory using graph database"""

            def __init__(self, neo4j_driver):
                self.driver = neo4j_driver

            def add_entity(self, agent_id: str, entity: dict):
                """Add entity node"""

                with self.driver.session() as session:
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.name = $name,
                            e.type = $type,
                            e.agent_id = $agent_id,
                            e.created_at = datetime()
                    """, {
                        "id": entity["id"],
                        "name": entity["name"],
                        "type": entity["type"],
                        "agent_id": agent_id
                    })

            def add_relationship(
                self,
                entity1_id: str,
                entity2_id: str,
                relationship_type: str,
                properties: dict = None
            ):
                """Add relationship between entities"""

                with self.driver.session() as session:
                    session.run(f"""
                        MATCH (e1:Entity {{id: $entity1_id}})
                        MATCH (e2:Entity {{id: $entity2_id}})
                        MERGE (e1)-[r:{relationship_type}]->(e2)
                        SET r += $properties,
                            r.created_at = datetime()
                    """, {
                        "entity1_id": entity1_id,
                        "entity2_id": entity2_id,
                        "properties": properties or {}
                    })

            def find_related(
                self,
                entity_id: str,
                relationship_types: list = None,
                max_depth: int = 2
            ) -> list:
                """Find related entities"""

                with self.driver.session() as session:
                    rel_filter = ""
                    if relationship_types:
                        rel_filter = f":{':'.join(relationship_types)}"

                    result = session.run(f"""
                        MATCH (e1:Entity {{id: $entity_id}})
                        MATCH path = (e1)-[r{rel_filter}*1..{max_depth}]-(e2)
                        RETURN e2, r, length(path) as distance
                        ORDER BY distance
                    """, {"entity_id": entity_id})

                    return [
                        {
                            "entity": dict(record["e2"]),
                            "relationships": [dict(rel) for rel in record["r"]],
                            "distance": record["distance"]
                        }
                        for record in result
                    ]

            def query_graph(self, agent_id: str, query: str) -> str:
                """Natural language query over knowledge graph"""

                # Convert NL to Cypher using LLM
                cypher_query = self._nl_to_cypher(query, agent_id)

                # Execute query
                with self.driver.session() as session:
                    result = session.run(cypher_query)
                    return [dict(record) for record in result]

            def _nl_to_cypher(self, nl_query: str, agent_id: str) -> str:
                """Convert natural language to Cypher query"""
                # Use LLM to generate Cypher from natural language
                # (Implementation details omitted)
                pass
        ```

        ---

        ## Unified Memory Manager

        ```python
        class MemoryManager:
            """Unified interface for all memory types"""

            def __init__(
                self,
                short_term: ShortTermMemory,
                vector: VectorMemory,
                graph: GraphMemory,
                episodic: EpisodicMemory
            ):
                self.short_term = short_term
                self.vector = vector
                self.graph = graph
                self.episodic = episodic

            def remember(
                self,
                agent_id: str,
                content: str,
                memory_type: str = "auto",
                metadata: dict = None
            ):
                """Store memory in appropriate storage"""

                if memory_type == "auto":
                    # Decide based on content
                    memory_type = self._classify_memory(content)

                if memory_type == "short_term":
                    self.short_term.add_message(agent_id, {
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif memory_type == "long_term":
                    # Store in both vector and graph
                    self.vector.store(agent_id, content, metadata)

                    # Extract entities and relationships
                    entities = self._extract_entities(content)
                    for entity in entities:
                        self.graph.add_entity(agent_id, entity)

                    relationships = self._extract_relationships(content)
                    for rel in relationships:
                        self.graph.add_relationship(
                            rel["entity1_id"],
                            rel["entity2_id"],
                            rel["type"]
                        )

            def recall(
                self,
                agent_id: str,
                query: str,
                memory_types: list = None,
                limit: int = 10
            ) -> list:
                """Retrieve relevant memories"""

                results = []

                # Search each memory type
                if not memory_types or "short_term" in memory_types:
                    short_term = self.short_term.get_context(agent_id, limit=5)
                    results.extend([
                        {"type": "short_term", "content": msg}
                        for msg in short_term
                    ])

                if not memory_types or "long_term" in memory_types:
                    vector_results = self.vector.search(agent_id, query, limit=limit)
                    results.extend([
                        {"type": "long_term_vector", **result}
                        for result in vector_results
                    ])

                if not memory_types or "graph" in memory_types:
                    graph_results = self.graph.query_graph(agent_id, query)
                    results.extend([
                        {"type": "long_term_graph", **result}
                        for result in graph_results
                    ])

                # Sort by relevance
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

                return results[:limit]
        ```

    === "🤝 Multi-Agent Orchestration"

        ## The Challenge

        Complex tasks require **multiple specialized agents working together**. Challenges include:

        - Coordinating agent execution (sequential, parallel, hierarchical)
        - Sharing information between agents
        - Managing dependencies and task distribution
        - Preventing conflicts and race conditions

        ---

        ## Orchestration Patterns

        ### Pattern 1: Sequential (Pipeline)

        **Concept:** Agents execute one after another, each using previous results.

        ```
        User Task: "Research quantum computing and write a report"

        ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
        │  Researcher  │ --> │   Analyzer   │ --> │    Writer    │
        │    Agent     │     │    Agent     │     │    Agent     │
        └──────────────┘     └──────────────┘     └──────────────┘
             Search              Evaluate             Generate
            sources             findings              report
        ```

        ```python
        class SequentialOrchestrator:
            """Execute agents in sequence"""

            def execute(self, task: dict) -> dict:
                """Run agents sequentially"""

                agents = task["agents"]
                context = {"task": task["description"]}

                for agent_config in agents:
                    agent = self.load_agent(agent_config["agent_id"])

                    # Execute agent with context from previous agents
                    result = agent.execute(
                        input=agent_config["task"],
                        context=context
                    )

                    # Add result to shared context
                    context[agent_config["role"]] = result

                return context
        ```

        ---

        ### Pattern 2: Parallel (Map-Reduce)

        **Concept:** Multiple agents work simultaneously, results aggregated.

        ```
        User Task: "Analyze sentiment from 10 news sources"

                            ┌──────────────┐
                        ┌──>│  Analyzer 1  │──┐
                        │   └──────────────┘  │
        ┌───────────┐   │   ┌──────────────┐  │   ┌────────────┐
        │ Splitter  │───┼──>│  Analyzer 2  │──┼──>│ Aggregator │
        │   Agent   │   │   └──────────────┘  │   │   Agent    │
        └───────────┘   │   ┌──────────────┐  │   └────────────┘
                        └──>│  Analyzer 3  │──┘
                            └──────────────┘
        ```

        ```python
        class ParallelOrchestrator:
            """Execute agents in parallel"""

            async def execute(self, task: dict) -> dict:
                """Run agents concurrently"""

                agents = task["agents"]

                # Create tasks for all agents
                tasks = [
                    self._execute_agent(agent_config)
                    for agent_config in agents
                ]

                # Execute in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Aggregate results
                aggregated = self._aggregate_results(results, task)

                return aggregated

            async def _execute_agent(self, agent_config: dict) -> dict:
                """Execute single agent"""
                agent = self.load_agent(agent_config["agent_id"])

                result = await agent.execute_async(
                    input=agent_config["task"],
                    context=agent_config.get("context")
                )

                return {
                    "agent_id": agent_config["agent_id"],
                    "result": result
                }

            def _aggregate_results(self, results: list, task: dict) -> dict:
                """Combine results from parallel agents"""

                # Use aggregator agent or simple combination
                if "aggregator" in task:
                    aggregator = self.load_agent(task["aggregator"])
                    return aggregator.execute(
                        input=f"Combine these results: {results}"
                    )

                # Simple concatenation
                return {
                    "results": results,
                    "combined": "\n\n".join([r["result"] for r in results])
                }
        ```

        ---

        ### Pattern 3: Hierarchical (Manager-Worker)

        **Concept:** Manager agent delegates subtasks to worker agents.

        ```
        User Task: "Build a comprehensive market analysis"

                        ┌─────────────────┐
                        │  Manager Agent  │
                        │  (Coordinator)  │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  Data Agent  │ │ Analysis Agent│ │ Report Agent │
            └──────────────┘ └──────────────┘ └──────────────┘
        ```

        ```python
        class HierarchicalOrchestrator:
            """Manager agent delegates to worker agents"""

            def execute(self, task: dict) -> dict:
                """Execute hierarchical task"""

                manager = self.load_agent(task["manager_agent_id"])

                # Manager creates plan
                plan = manager.execute(
                    input=f"Create a plan to: {task['description']}",
                    tools=["create_subtask", "delegate_task"]
                )

                # Execute subtasks
                results = []
                for subtask in plan["subtasks"]:
                    # Manager decides which worker
                    worker = self._select_worker(subtask)

                    # Delegate to worker
                    result = worker.execute(
                        input=subtask["description"],
                        context={"parent_task": task}
                    )

                    results.append({
                        "subtask": subtask,
                        "result": result
                    })

                # Manager synthesizes results
                final_result = manager.execute(
                    input=f"Synthesize results: {results}",
                    context={"original_task": task}
                )

                return final_result
        ```

        ---

        ## Agent Communication

        ```python
        class MessageBus:
            """Pub/sub message bus for agent communication"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.subscribers = {}

            def publish(
                self,
                channel: str,
                message: dict,
                sender_agent_id: str
            ):
                """Publish message to channel"""

                message_with_metadata = {
                    "sender": sender_agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": message
                }

                self.redis.publish(
                    channel,
                    json.dumps(message_with_metadata)
                )

            def subscribe(
                self,
                agent_id: str,
                channels: list,
                callback: callable
            ):
                """Subscribe agent to channels"""

                pubsub = self.redis.pubsub()
                pubsub.subscribe(channels)

                self.subscribers[agent_id] = pubsub

                # Listen for messages in background thread
                thread = threading.Thread(
                    target=self._listen,
                    args=(pubsub, callback)
                )
                thread.daemon = True
                thread.start()

            def _listen(self, pubsub, callback):
                """Listen for messages"""
                for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        callback(data)

            def send_direct(
                self,
                from_agent_id: str,
                to_agent_id: str,
                message: dict
            ):
                """Send direct message to specific agent"""

                channel = f"agent:{to_agent_id}:inbox"
                self.publish(channel, message, from_agent_id)

        # Usage
        message_bus = MessageBus(redis_client)

        # Agent 1 subscribes
        def handle_message(message):
            print(f"Received: {message}")

        message_bus.subscribe(
            agent_id="agent_1",
            channels=["research_results"],
            callback=handle_message
        )

        # Agent 2 publishes
        message_bus.publish(
            channel="research_results",
            message={"topic": "quantum computing", "findings": [...]},
            sender_agent_id="agent_2"
        )
        ```

        ---

        ## Shared Memory for Multi-Agent

        ```python
        class SharedMemory:
            """Shared memory space for agent collaboration"""

            def __init__(self, redis_client):
                self.redis = redis_client

            def write(
                self,
                task_id: str,
                key: str,
                value: Any,
                agent_id: str,
                ttl: int = 3600
            ):
                """Write to shared memory"""

                memory_key = f"task:{task_id}:shared:{key}"

                data = {
                    "value": value,
                    "written_by": agent_id,
                    "written_at": datetime.utcnow().isoformat()
                }

                self.redis.setex(
                    memory_key,
                    ttl,
                    json.dumps(data)
                )

            def read(self, task_id: str, key: str) -> Any:
                """Read from shared memory"""

                memory_key = f"task:{task_id}:shared:{key}"

                data = self.redis.get(memory_key)
                if data:
                    return json.loads(data)
                return None

            def read_all(self, task_id: str) -> dict:
                """Read all shared memory for task"""

                pattern = f"task:{task_id}:shared:*"
                keys = self.redis.keys(pattern)

                result = {}
                for key in keys:
                    key_name = key.decode().split(':')[-1]
                    data = self.redis.get(key)
                    result[key_name] = json.loads(data)

                return result
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling an agent platform from prototype (10 agents) to production (100K agents) requires addressing bottlenecks in LLM inference, tool execution, memory operations, and orchestration.

    **Scaling philosophy:**

    1. **Optimize LLM calls** - Cache prompts, batch requests, use smaller models when possible
    2. **Parallelize tool execution** - Run independent tools concurrently
    3. **Distributed memory** - Shard vector DB, cache hot memories
    4. **Agent pooling** - Pre-warm agents, reuse initialized instances

    ---

    ## Bottleneck Identification

    | Component | Current Capacity | Required Capacity | Bottleneck? | Solution |
    |-----------|-----------------|------------------|------------|----------|
    | **LLM Gateway** | 500 req/sec | 520 req/sec | 🟡 Near limit | Add caching, batching, multiple providers |
    | **Tool Executors** | 100 concurrent | 350 concurrent | ✅ Need scaling | Agent pools, container reuse |
    | **Vector DB** | 1K queries/sec | 580 queries/sec | ❌ Not bottleneck | Current capacity sufficient |
    | **Graph DB** | 500 queries/sec | 100 queries/sec | ❌ Not bottleneck | Current capacity sufficient |
    | **Message Bus** | 10K msg/sec | 35 msg/sec | ❌ Not bottleneck | Redis easily handles this |

    **Critical bottleneck: Tool execution sandboxing**

    Spinning up Docker containers takes 500ms. At 350 tool calls/sec, we need 175 concurrent containers. Solution: Pre-warm container pools.

    ---

    ## Optimization Strategies

    ### 1. LLM Call Optimization

    **Problem:** LLM calls are expensive ($0.01 per 1K tokens) and slow (1-3s latency).

    **Solution: Prompt caching + batching**

    ```python
    class OptimizedLLMGateway:
        """LLM gateway with aggressive caching and batching"""

        def __init__(self):
            self.prompt_cache = {}  # Cache common prompts
            self.request_queue = []  # Batch requests
            self.batch_size = 10
            self.batch_timeout = 100  # ms

        async def complete(self, prompt: str, **kwargs) -> str:
            """Complete with caching"""

            # Check cache
            cache_key = self._cache_key(prompt, kwargs)
            if cache_key in self.prompt_cache:
                return self.prompt_cache[cache_key]

            # Add to batch queue
            future = asyncio.Future()
            self.request_queue.append((prompt, kwargs, future))

            # Process batch if full or timeout
            if len(self.request_queue) >= self.batch_size:
                await self._process_batch()
            else:
                # Set timer for batch timeout
                asyncio.create_task(self._batch_timeout())

            result = await future

            # Cache result
            self.prompt_cache[cache_key] = result

            return result

        async def _process_batch(self):
            """Process batch of requests"""
            if not self.request_queue:
                return

            batch = self.request_queue[:self.batch_size]
            self.request_queue = self.request_queue[self.batch_size:]

            # Send batch to LLM
            prompts = [req[0] for req in batch]
            results = await self.llm.batch_complete(prompts)

            # Resolve futures
            for (prompt, kwargs, future), result in zip(batch, results):
                future.set_result(result)
    ```

    **Improvements:**
    - Prompt cache reduces redundant calls by 30-40%
    - Batching improves throughput by 3-5x
    - Use cheaper models (GPT-3.5) for simple reasoning steps

    ---

    ### 2. Tool Execution Pooling

    **Problem:** Container spin-up takes 500ms, dominates tool execution time.

    **Solution: Pre-warmed container pools**

    ```python
    class ContainerPool:
        """Pool of pre-warmed containers for tool execution"""

        def __init__(self, pool_size=50):
            self.pool = queue.Queue(maxsize=pool_size)
            self.docker = docker.from_env()

            # Pre-warm pool
            for _ in range(pool_size):
                container = self._create_container()
                self.pool.put(container)

        def _create_container(self):
            """Create and start container"""
            container = self.docker.containers.create(
                image="agent-tool-executor:latest",
                command="sleep infinity",  # Keep alive
                mem_limit="512m",
                detach=True
            )
            container.start()
            return container

        def get_container(self, timeout=5):
            """Get container from pool"""
            try:
                return self.pool.get(timeout=timeout)
            except queue.Empty:
                # Pool exhausted, create new one
                return self._create_container()

        def return_container(self, container):
            """Return container to pool"""
            try:
                # Reset container state
                container.exec_run("rm -rf /tmp/*")
                self.pool.put_nowait(container)
            except queue.Full:
                # Pool full, destroy container
                container.stop()
                container.remove()

        def execute_tool(self, tool_code: str, tool_input: dict):
            """Execute tool using pooled container"""
            container = self.get_container()

            try:
                # Execute tool
                result = container.exec_run(
                    f"python -c '{tool_code}' '{json.dumps(tool_input)}'",
                    timeout=5
                )
                return result.output.decode()
            finally:
                # Always return to pool
                self.return_container(container)
    ```

    **Improvements:**
    - Reduces tool execution latency from 1s to 100ms (10x faster)
    - Supports 350 concurrent tool calls with 50 pooled containers
    - Container reuse reduces memory pressure

    ---

    ### 3. Memory Sharding

    **Problem:** Single vector DB instance can't handle 580 searches/sec + writes.

    **Solution: Shard by agent_id**

    ```python
    class ShardedVectorMemory:
        """Sharded vector memory for horizontal scaling"""

        def __init__(self, num_shards=10):
            self.num_shards = num_shards
            self.shards = [
                PineconeClient(index=f"memory-shard-{i}")
                for i in range(num_shards)
            ]

        def _get_shard(self, agent_id: str) -> PineconeClient:
            """Determine shard for agent"""
            shard_id = hash(agent_id) % self.num_shards
            return self.shards[shard_id]

        def store(self, agent_id: str, content: str, metadata: dict):
            """Store in appropriate shard"""
            shard = self._get_shard(agent_id)
            embedding = self.embedding_model.embed(content)

            shard.upsert({
                "id": f"mem_{uuid.uuid4()}",
                "vector": embedding,
                "metadata": {
                    "agent_id": agent_id,
                    "content": content,
                    **metadata
                }
            })

        def search(self, agent_id: str, query: str, limit: int = 10):
            """Search in appropriate shard"""
            shard = self._get_shard(agent_id)
            query_embedding = self.embedding_model.embed(query)

            return shard.query(
                vector=query_embedding,
                top_k=limit,
                filter={"agent_id": agent_id}
            )
    ```

    **Improvements:**
    - 10 shards = 10x capacity (5,800 queries/sec)
    - Each shard independent (no cross-shard queries needed)
    - Easily add more shards as needed

    ---

    ### 4. Agent Instance Pooling

    **Problem:** Initializing agent (loading tools, memory) takes 2-3 seconds.

    **Solution: Keep agents warm in memory**

    ```python
    class AgentPool:
        """Pool of initialized agents"""

        def __init__(self, max_size=1000):
            self.pool = {}  # agent_id -> agent instance
            self.lru = OrderedDict()  # LRU eviction
            self.max_size = max_size

        def get_agent(self, agent_id: str):
            """Get or create agent"""

            if agent_id in self.pool:
                # Move to end (most recently used)
                self.lru.move_to_end(agent_id)
                return self.pool[agent_id]

            # Load agent
            agent = self._load_agent(agent_id)

            # Add to pool
            self.pool[agent_id] = agent
            self.lru[agent_id] = time.time()

            # Evict LRU if full
            if len(self.pool) > self.max_size:
                lru_agent_id = next(iter(self.lru))
                del self.pool[lru_agent_id]
                del self.lru[lru_agent_id]

            return agent

        def _load_agent(self, agent_id: str):
            """Load agent from database"""
            # Fetch agent definition
            agent_data = db.query("SELECT * FROM agents WHERE id = %s", (agent_id,))

            # Load tools
            tools = self._load_tools(agent_data["tools"])

            # Initialize memory
            memory = MemoryManager(...)

            # Create agent
            return ReactAgent(
                tools=tools,
                memory=memory,
                config=agent_data["config"]
            )
    ```

    **Improvements:**
    - Agent execution starts immediately (no init overhead)
    - Supports 1,440 concurrent agents with 1,000 pool size
    - LRU eviction keeps hot agents in memory

    ---

    ## Monitoring & Metrics

    **Critical metrics for agent platforms:**

    | Metric | Target | Alert Threshold | Why It Matters |
    |--------|--------|-----------------|----------------|
    | **Agent Success Rate** | > 70% | < 60% | Core platform reliability |
    | **Avg Iterations** | 5-8 iterations | > 12 | Agent efficiency, prompt quality |
    | **Tool Execution Time** | < 1s p95 | > 3s | Sandbox performance |
    | **LLM Latency** | < 2s p95 | > 5s | User experience |
    | **Memory Query Latency** | < 100ms p95 | > 500ms | Context retrieval speed |
    | **Token Usage** | $0.05 per execution | > $0.15 | Cost control |
    | **Container Pool Exhaustion** | < 5% | > 20% | Scaling indicator |

    **Observability tools:**
    - **Jaeger**: Distributed tracing for agent execution flows
    - **Prometheus**: Metrics for tool calls, LLM requests, memory ops
    - **Grafana**: Dashboards for agent performance
    - **Datadog APM**: End-to-end execution visibility

=== "📝 Summary & Tips"

    ## Architecture Summary

    **Core Components:**

    | Component | Purpose | Technology | Quantity |
    |-----------|---------|------------|----------|
    | **Agent Engine** | ReAct loop execution | Python/LangChain | 20 servers |
    | **Tool Registry** | Tool discovery and management | PostgreSQL + embeddings | 1 primary, 2 replicas |
    | **Tool Executor** | Sandboxed tool execution | Docker/gVisor pools | 50 pooled containers |
    | **LLM Gateway** | Model routing and caching | Multi-provider (OpenAI, Anthropic) | 10 servers |
    | **Vector DB** | Long-term semantic memory | Pinecone/Weaviate (sharded) | 10 shards |
    | **Graph DB** | Entity relationship memory | Neo4j | 1 cluster (3 nodes) |
    | **Message Bus** | Multi-agent communication | Redis Pub/Sub | 1 cluster |

    ---

    ## Capacity Handled

    | Metric | Capacity | Headroom |
    |--------|----------|----------|
    | **Agent executions** | 1M executions/day (~12 QPS) | 5x capacity (60 QPS) |
    | **Tool calls** | 10M calls/day (~116 QPS) | 3x capacity (350 QPS) |
    | **LLM requests** | 15M requests/day (~174 QPS) | 3x capacity (520 QPS) |
    | **Memory operations** | 70M ops/day (580 read QPS, 230 write QPS) | 10x capacity |
    | **Concurrent agents** | 4,320 peak concurrent | Pool size: 1,000 warm agents |

    ---

    ## Key Design Decisions

    1. **ReAct over simple prompting** - Enables reasoning, tool use, and self-correction
    2. **Sandboxed execution** - Security over raw speed for tool execution
    3. **Multi-layer memory** - Short-term (Redis) + Long-term (Vector + Graph) + Episodic (MongoDB)
    4. **Vector-based tool discovery** - Semantic search over keyword matching
    5. **Message bus for multi-agent** - Async pub/sub over direct calls
    6. **Container pooling** - Pre-warmed containers reduce latency from 1s to 100ms
    7. **LLM provider abstraction** - Fallback between OpenAI/Anthropic/local models

    ---

    ## Interview Tips

    ### What Interviewers Look For

    ✅ **Understand agent architectures** - ReAct vs Chain-of-Thought vs Tree-of-Thought
    ✅ **Explain tool calling** - OpenAI function calling vs Anthropic tools format
    ✅ **Memory design** - Why vector + graph? When to use episodic memory?
    ✅ **Sandboxing rationale** - Security vs performance trade-off
    ✅ **Multi-agent patterns** - Sequential vs parallel vs hierarchical orchestration
    ✅ **Scaling challenges** - LLM calls, tool execution, memory queries
    ✅ **Cost optimization** - Prompt caching, model selection, result reuse

    ---

    ### Common Follow-up Questions

    | Question | Key Points to Cover |
    |----------|-------------------|
    | **"How does ReAct differ from Chain-of-Thought?"** | ReAct adds tool calling to CoT reasoning, enables interaction with external systems |
    | **"How do you prevent infinite loops in agent execution?"** | Max iterations limit (15), timeout per step (30s), cost budget per execution |
    | **"How do agents share context in multi-agent systems?"** | Shared memory (Redis), message bus (pub/sub), episodic memory (execution logs) |
    | **"How do you handle tool execution failures?"** | Retry with exponential backoff, fallback to alternative tools, circuit breaker pattern |
    | **"Why use both vector and graph databases?"** | Vector for semantic search, graph for entity relationships and complex queries |
    | **"How do you optimize LLM costs?"** | Prompt caching (30-40% savings), smaller models for simple steps, result caching |
    | **"How do you evaluate agent quality?"** | Success rate, avg iterations, task completion time, human eval on sample tasks |
    | **"How do you debug agents when they fail?"** | Full execution traces, step-by-step logging, replay execution, inspect tool I/O |

    ---

    ### Things to Mention

    **ReAct Pattern:**
    - Industry standard for autonomous agents (LangChain, AutoGPT)
    - Enables self-correction and multi-step reasoning
    - Thought → Action → Observation cycle

    **Sandboxing:**
    - Docker/gVisor isolation prevents malicious code
    - Network restrictions, memory limits, CPU quotas
    - Pre-warmed pools reduce 500ms overhead

    **Memory Architecture:**
    - Short-term: Last 20 messages in Redis
    - Long-term vector: Semantic search over past conversations
    - Long-term graph: Entity relationships and knowledge graph
    - Episodic: Full execution traces for learning

    **Multi-Agent:**
    - Sequential: Pipeline pattern (research → analyze → write)
    - Parallel: Map-reduce pattern (distribute → aggregate)
    - Hierarchical: Manager delegates to specialized workers

    ---

    ## Related Problems

    | Problem | Similarity | Key Differences |
    |---------|------------|-----------------|
    | **ChatGPT System** | LLM-powered, streaming responses | No tool calling, no multi-step reasoning, simpler memory |
    | **RAG System** | Vector memory, semantic search | No agent loop, no tool execution, retrieval only |
    | **Code Execution Platform** | Sandboxing, isolated execution | No LLM, no reasoning, simpler orchestration |
    | **Workflow Orchestration (Airflow)** | Task dependencies, DAG execution | No LLM, deterministic (not agentic), no reasoning |
    | **RPA Platform** | Automate tasks, tool execution | Rule-based (not AI), no reasoning, no learning |

    ---

    ## Real-World Examples

    **LangChain Agents:**
    ```python
    from langchain.agents import initialize_agent, Tool
    from langchain.llms import OpenAI

    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="Search the web for information"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Perform mathematical calculations"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=OpenAI(temperature=0),
        agent="zero-shot-react-description",
        verbose=True
    )

    result = agent.run("What is the population of France times 2?")
    ```

    **AutoGPT Architecture:**
    - Long-term memory (Pinecone)
    - Short-term memory (conversation context)
    - Tool execution (web browsing, file operations)
    - Self-reflection and planning

    **CrewAI Multi-Agent:**
    ```python
    from crewai import Agent, Task, Crew

    researcher = Agent(
        role='Researcher',
        goal='Research quantum computing breakthroughs',
        tools=[web_search, web_scraper]
    )

    writer = Agent(
        role='Writer',
        goal='Write comprehensive report',
        tools=[document_writer]
    )

    task1 = Task(description='Research quantum computing', agent=researcher)
    task2 = Task(description='Write report', agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
    result = crew.kickoff()
    ```

    ---

    ## Next Steps

    **After mastering AI Agent Platform:**

    1. **RAG System** - Deep dive into retrieval-augmented generation
    2. **Model Serving** - Deploy and scale LLM inference
    3. **Feature Store** - ML infrastructure for model inputs
    4. **Recommendation System** - Personalization and ranking

    **Practice variations:**

    - Add human-in-the-loop approval for sensitive actions
    - Implement agent learning from past executions
    - Build visual agent execution debugger
    - Add cost budget enforcement per execution
    - Implement agent benchmarking framework (SWE-bench, GAIA)
    - Support voice/video agent interactions

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** LangChain, AutoGPT, BabyAGI, SuperAGI, CrewAI, Anthropic, Microsoft

---

*This template applies to similar problems: Autonomous AI Agents, Multi-Agent Systems, Tool-Using AI, Agentic Workflows*
