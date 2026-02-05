# Design RAG System (Retrieval Augmented Generation)

A scalable retrieval augmented generation system that combines document retrieval with large language models to provide accurate, contextual, and cited responses based on enterprise knowledge bases.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M documents, 100K queries/day, 50B tokens processed/month, 10K concurrent users |
| **Key Challenges** | Document chunking, embedding quality, retrieval accuracy, context window limits, hallucination prevention, real-time updates |
| **Core Concepts** | Vector embeddings, semantic search, hybrid retrieval, reranking, prompt engineering, citation tracking, conversation memory |
| **Companies** | OpenAI, Anthropic, LangChain, LlamaIndex, Enterprise AI, Cohere, Pinecone, Weaviate |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Document Ingestion** | Upload, parse, and index documents (PDF, DOCX, TXT, HTML, MD) | P0 (Must have) |
    | **Semantic Search** | Find relevant document chunks using vector similarity | P0 (Must have) |
    | **LLM Response Generation** | Generate contextual answers using retrieved content | P0 (Must have) |
    | **Citation Tracking** | Provide source references for every statement | P0 (Must have) |
    | **Multi-turn Conversations** | Maintain context across conversation turns | P0 (Must have) |
    | **Hybrid Retrieval** | Combine semantic + keyword search for better results | P1 (Should have) |
    | **Reranking** | Re-score retrieved chunks for better relevance | P1 (Should have) |
    | **Real-time Updates** | Incremental indexing for new/updated documents | P1 (Should have) |
    | **Query Decomposition** | Break complex queries into sub-queries | P2 (Nice to have) |
    | **Multi-document Reasoning** | Synthesize information from multiple sources | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Document creation/editing interface
    - User authentication/authorization (assume separate system)
    - Document access control (assume handled elsewhere)
    - LLM fine-tuning infrastructure
    - Content moderation
    - Multi-language translation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 3s p95 for end-to-end response | Users expect fast answers (retrieval + LLM generation) |
    | **Retrieval Accuracy** | Recall@10 > 90%, Precision@10 > 80% | Must find relevant context to avoid hallucinations |
    | **Availability** | 99.9% uptime | Critical for enterprise applications |
    | **Freshness** | < 5 minutes for document updates | New content should be queryable quickly |
    | **Citation Accuracy** | 100% attribution to source documents | Essential for trust and verification |
    | **Scalability** | Support 10x traffic spikes | Handle batch processing and peak usage |
    | **Cost Efficiency** | < $0.10 per query (LLM + embedding + infra) | LLM costs dominate, need optimization |
    | **Context Relevance** | > 95% relevant chunks in top-K | Irrelevant context increases hallucinations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 50K
    Monthly Active Users (MAU): 150K

    Queries:
    - Queries per DAU: 2 queries/day
    - Daily queries: 50K × 2 = 100K queries/day
    - Query QPS: 100K / 86,400 = ~1.2 req/sec
    - Peak QPS: 10x average = ~12 req/sec

    Document ingestion:
    - New documents: 10K documents/day (constant ingestion)
    - Document update rate: 5% daily (500K updates/day)
    - Ingestion QPS: 10K / 86,400 = ~0.12 docs/sec
    - Batch processing: Often done in batches (e.g., 1K docs every 2 hours)

    Embedding generation:
    - Query embeddings: 100K/day
    - Document chunk embeddings: 10K docs × 50 chunks = 500K chunks/day
    - Total embeddings: 600K/day = ~7 embeddings/sec

    LLM tokens:
    - Avg tokens per query: 2K (context) + 500 (response) = 2.5K tokens
    - Daily tokens: 100K queries × 2.5K = 250M tokens/day
    - Monthly tokens: 250M × 30 = 7.5B tokens/month
    - Cost: 7.5B tokens × $0.002/1K = $15K/month (GPT-4 class)

    Read/Write ratio: 100:1 (mostly read for retrieval)
    ```

    ### Storage Estimates

    ```
    Document storage:
    - Total documents: 10M documents
    - Avg document size: 500 KB
    - Raw storage: 10M × 500 KB = 5 TB

    Document chunks:
    - Avg chunks per document: 50 chunks (1000-token chunks with 200 overlap)
    - Total chunks: 10M × 50 = 500M chunks
    - Chunk metadata: 500 bytes (doc_id, chunk_id, position, hash)
    - Metadata storage: 500M × 500 bytes = 250 GB

    Vector embeddings:
    - Embedding model: 768-dim (sentence-transformers) or 1536-dim (OpenAI)
    - Embedding size: 1536 floats × 4 bytes = 6 KB per embedding
    - Total embeddings: 500M chunks × 6 KB = 3 TB

    Conversation history:
    - Active conversations: 10K concurrent
    - Avg conversation: 10 turns × 3 KB = 30 KB
    - Conversation storage: 10K × 30 KB = 300 MB
    - Historical: 1M conversations × 30 KB = 30 GB

    Pre-computed indexes:
    - Vector index (HNSW): ~1.5x raw embeddings = 4.5 TB
    - Inverted index (BM25): 500M chunks × 1 KB = 500 GB

    Cache:
    - Hot query embeddings: 10K queries × 6 KB = 60 MB
    - Popular responses: 1K responses × 5 KB = 5 MB

    Total: 5 TB (docs) + 250 GB (metadata) + 3 TB (embeddings) + 4.5 TB (indexes) + 500 GB (inverted) + 30 GB (conversations) ≈ 13.3 TB
    ```

    ### Bandwidth Estimates

    ```
    Query API:
    - 1.2 req/sec × 500 bytes (query) = 600 bytes/sec (ingress)
    - 1.2 req/sec × 5 KB (response + citations) = 6 KB/sec (egress)

    Retrieval:
    - 1.2 req/sec × 10 chunks × 1 KB = 12 KB/sec (retrieved content)

    LLM API calls:
    - 1.2 req/sec × 2.5 KB (prompt + context) = 3 KB/sec (to LLM)
    - 1.2 req/sec × 1.5 KB (response) = 1.8 KB/sec (from LLM)

    Document ingestion:
    - 0.12 docs/sec × 500 KB = 60 KB/sec

    Embedding generation:
    - 7 embeddings/sec × 6 KB = 42 KB/sec

    Total ingress: ~64 KB/sec
    Total egress: ~20 KB/sec
    ```

    ### Memory Estimates (Caching)

    ```
    Vector index cache (HNSW):
    - Hot vectors: 10M chunks (top 2% accessed frequently)
    - 10M × 6 KB = 60 GB

    Query embedding cache:
    - 10K hot queries × 6 KB = 60 MB

    Document chunk cache:
    - 1M hot chunks × 1 KB = 1 GB

    LLM response cache:
    - 1K common queries × 5 KB = 5 MB

    Conversation context cache:
    - 10K active sessions × 30 KB = 300 MB

    Total cache: 60 GB + 60 MB + 1 GB + 5 MB + 300 MB ≈ 62 GB
    ```

    ---

    ## Key Assumptions

    1. Average document size: 500 KB (~125K tokens)
    2. Chunk size: 1000 tokens with 200-token overlap (~80% of doc length becomes chunks)
    3. Retrieval: Top-10 chunks per query
    4. LLM context window: 8K tokens (allows ~6K context + 2K response)
    5. Embedding latency: 50ms for query, 100ms per document (batched)
    6. LLM latency: 2-3 seconds for response generation
    7. Vector search latency: 10-50ms for ANN search
    8. 80% of queries are unique, 20% are repeated (cacheable)
    9. Documents updated 5% daily, requiring re-embedding
    10. Conversation sessions average 5 turns before completion

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separate ingestion and retrieval:** Async document processing, fast query serving
    2. **Hybrid retrieval:** Combine dense (semantic) + sparse (keyword) search
    3. **Multi-stage retrieval:** Fast candidate generation → reranking → final selection
    4. **Citation tracking:** Maintain provenance from chunks to final response
    5. **Stateful conversations:** Track context across multi-turn dialogues
    6. **Incremental indexing:** Real-time updates without full reindexing
    7. **Cost optimization:** Cache embeddings and responses, batch LLM calls

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            User[User/Application]
            API_Gateway[API Gateway<br/>Rate limiting, auth]
        end

        subgraph "Query Processing Pipeline"
            Query_API[Query API<br/>Handle user queries]
            Query_Analyzer[Query Analyzer<br/>Intent, decomposition]
            Embedding_Service[Embedding Service<br/>Generate query embeddings]
            Conversation_Manager[Conversation Manager<br/>Multi-turn context]
        end

        subgraph "Retrieval Engine"
            Retriever[Retrieval Orchestrator<br/>Coordinate search]
            Vector_Search[Vector Search<br/>Semantic similarity]
            Keyword_Search[Keyword Search<br/>BM25/lexical]
            Hybrid_Fusion[Hybrid Fusion<br/>Combine scores]
            Reranker[Reranker<br/>Cross-encoder scoring]
        end

        subgraph "Generation Pipeline"
            Context_Builder[Context Builder<br/>Assemble prompt]
            LLM_Gateway[LLM Gateway<br/>Load balancing]
            LLM_Cache[LLM Response Cache<br/>Redis]
            Citation_Tracker[Citation Tracker<br/>Source attribution]
        end

        subgraph "Document Ingestion Pipeline"
            Doc_Upload[Document Upload API]
            Doc_Parser[Document Parser<br/>PDF, DOCX, HTML]
            Chunker[Chunking Service<br/>Recursive/semantic split]
            Metadata_Extractor[Metadata Extractor<br/>Title, author, date]
            Embedding_Generator[Embedding Generator<br/>Batch processing]
        end

        subgraph "Storage Layer"
            Doc_Store[(Document Store<br/>S3/MinIO)]
            Vector_DB[(Vector Database<br/>Pinecone/Weaviate)]
            Metadata_DB[(Metadata DB<br/>PostgreSQL)]
            Search_Index[(Search Index<br/>Elasticsearch)]
            Conversation_DB[(Conversation Store<br/>DynamoDB)]
        end

        subgraph "LLM Services"
            OpenAI[OpenAI API<br/>GPT-4]
            Anthropic[Anthropic API<br/>Claude]
            Local_LLM[Local LLM<br/>Llama/Mistral]
        end

        subgraph "Monitoring & Optimization"
            Metrics[Metrics Service<br/>Latency, accuracy]
            A_B_Testing[A/B Testing<br/>Retrieval strategies]
            Cost_Monitor[Cost Monitor<br/>LLM usage tracking]
        end

        subgraph "Caching Layers"
            Embedding_Cache[Embedding Cache<br/>Redis]
            Query_Cache[Query Cache<br/>Redis]
        end

        User --> API_Gateway
        API_Gateway --> Query_API
        API_Gateway --> Doc_Upload

        Query_API --> Query_Analyzer
        Query_API --> Conversation_Manager
        Query_Analyzer --> Embedding_Service
        Embedding_Service --> Embedding_Cache

        Embedding_Service --> Retriever
        Retriever --> Vector_Search
        Retriever --> Keyword_Search
        Vector_Search --> Vector_DB
        Keyword_Search --> Search_Index
        Vector_Search --> Hybrid_Fusion
        Keyword_Search --> Hybrid_Fusion
        Hybrid_Fusion --> Reranker
        Reranker --> Metadata_DB

        Reranker --> Context_Builder
        Context_Builder --> LLM_Gateway
        Context_Builder --> Conversation_Manager
        LLM_Gateway --> LLM_Cache
        LLM_Gateway --> OpenAI
        LLM_Gateway --> Anthropic
        LLM_Gateway --> Local_LLM

        LLM_Gateway --> Citation_Tracker
        Citation_Tracker --> Query_API
        Citation_Tracker --> Conversation_DB

        Doc_Upload --> Doc_Parser
        Doc_Parser --> Chunker
        Doc_Parser --> Doc_Store
        Chunker --> Metadata_Extractor
        Metadata_Extractor --> Metadata_DB
        Chunker --> Embedding_Generator
        Embedding_Generator --> Vector_DB
        Embedding_Generator --> Search_Index

        Query_API --> Metrics
        Doc_Upload --> Metrics
        LLM_Gateway --> Cost_Monitor

        style API_Gateway fill:#e1f5ff
        style Vector_DB fill:#e8eaf6
        style Metadata_DB fill:#ffe1e1
        style Doc_Store fill:#f3e5f5
        style Embedding_Cache fill:#fff4e1
        style Query_Cache fill:#fff4e1
        style LLM_Cache fill:#fff4e1
        style OpenAI fill:#e8f5e9
        style Anthropic fill:#e8f5e9
        style Local_LLM fill:#e8f5e9
        style Conversation_DB fill:#ffe1e1
        style Search_Index fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Pinecone/Weaviate (Vector DB)** | Fast ANN search (< 50ms), billion-scale embeddings, built-in filtering | FAISS (not distributed), Elasticsearch (slower for dense vectors), Postgres pgvector (scaling limits) |
    | **Elasticsearch** | Fast keyword search (BM25), inverted index, full-text capabilities | PostgreSQL full-text (limited features), custom inverted index (complex) |
    | **Redis (Multi-layer)** | Fast caching (< 1ms) for embeddings, responses, conversation state | Memcached (limited features), no cache (latency too high) |
    | **PostgreSQL (Metadata)** | ACID transactions, relational data (docs, chunks, users), complex queries | MongoDB (weaker consistency), DynamoDB (limited query patterns) |
    | **S3/MinIO** | Cheap object storage for raw documents, versioning, durability | File system (no durability), database BLOB (expensive) |
    | **LangChain/LlamaIndex** | Pre-built RAG patterns, document loaders, chain orchestration | Custom implementation (reinventing wheel, more bugs) |

    **Key Trade-off:** We chose **hybrid retrieval** (semantic + keyword) over pure semantic search. While semantic search captures intent, keyword search ensures exact match recall (e.g., product names, technical terms).

    ---

    ## API Design

    ### 1. Query API (Ask a Question)

    **Request:**
    ```http
    POST /api/v1/query
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "query": "What are the benefits of microservices architecture?",
      "conversation_id": "conv_123",  // Optional: for multi-turn
      "filters": {
        "document_types": ["pdf", "md"],
        "date_range": {
          "start": "2024-01-01",
          "end": "2026-02-01"
        },
        "tags": ["architecture", "backend"]
      },
      "retrieval_config": {
        "top_k": 10,                   // Number of chunks to retrieve
        "rerank": true,                // Enable reranking
        "hybrid_alpha": 0.7,           // Semantic vs keyword weight (0-1)
        "min_similarity": 0.7          // Similarity threshold
      },
      "generation_config": {
        "model": "gpt-4",              // LLM to use
        "max_tokens": 500,
        "temperature": 0.3,
        "include_citations": true
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "answer": "Microservices architecture offers several key benefits:\n\n1. **Independent Scalability**: Each service can be scaled independently based on demand [1][2].\n\n2. **Technology Flexibility**: Teams can choose the best technology stack for each service [2][3].\n\n3. **Faster Development**: Smaller, focused codebases enable faster iteration [1].\n\n4. **Fault Isolation**: Failures in one service don't bring down the entire system [3].\n\nHowever, it also introduces complexity in deployment, monitoring, and inter-service communication [1][2].",
      "sources": [
        {
          "citation_id": "[1]",
          "document_id": "doc_789",
          "document_title": "Microservices Design Patterns",
          "chunk_id": "chunk_123",
          "content": "Microservices allow independent scaling of services based on specific demand patterns. Each service can be deployed and scaled independently without affecting others...",
          "page": 12,
          "relevance_score": 0.92,
          "url": "https://storage.example.com/docs/microservices-patterns.pdf"
        },
        {
          "citation_id": "[2]",
          "document_id": "doc_456",
          "document_title": "Cloud Native Architecture Guide",
          "chunk_id": "chunk_456",
          "content": "Technology flexibility is a major advantage of microservices. Teams can choose the most appropriate language, framework, and database for each service...",
          "page": 34,
          "relevance_score": 0.89,
          "url": "https://storage.example.com/docs/cloud-native.pdf"
        },
        {
          "citation_id": "[3]",
          "document_id": "doc_789",
          "document_title": "Microservices Design Patterns",
          "chunk_id": "chunk_789",
          "content": "Fault isolation is crucial for resilient systems. In microservices, a failure in one service is contained and doesn't cascade to other services...",
          "page": 45,
          "relevance_score": 0.87,
          "url": "https://storage.example.com/docs/microservices-patterns.pdf"
        }
      ],
      "conversation_id": "conv_123",
      "metadata": {
        "retrieved_chunks": 10,
        "reranked": true,
        "retrieval_latency_ms": 45,
        "llm_latency_ms": 2340,
        "total_latency_ms": 2410,
        "tokens_used": 2450,
        "cost_usd": 0.0049,
        "model": "gpt-4"
      },
      "query_id": "query_xyz789"
    }
    ```

    **Design Notes:**

    - Return citations inline with the response (trust and verifiability)
    - Support conversation context for multi-turn dialogues
    - Provide detailed metadata for monitoring and debugging
    - Support flexible filtering on metadata (dates, types, tags)
    - Enable A/B testing with configurable retrieval parameters

    ---

    ### 2. Document Upload API

    **Request:**
    ```http
    POST /api/v1/documents
    Content-Type: multipart/form-data
    Authorization: Bearer <token>

    {
      "file": <binary file data>,
      "metadata": {
        "title": "Microservices Design Patterns",
        "author": "John Doe",
        "document_type": "pdf",
        "tags": ["architecture", "backend", "microservices"],
        "source_url": "https://example.com/doc.pdf",
        "language": "en",
        "access_level": "internal"
      },
      "processing_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_strategy": "recursive",  // recursive, semantic, fixed
        "extract_metadata": true,          // Extract title, headers, etc.
        "ocr_enabled": false               // For scanned PDFs
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "document_id": "doc_789",
      "status": "processing",
      "estimated_completion_seconds": 30,
      "chunks_estimated": 45
    }
    ```

    **Status Check:**
    ```http
    GET /api/v1/documents/doc_789/status

    {
      "document_id": "doc_789",
      "status": "completed",  // processing, completed, failed
      "chunks_created": 48,
      "embeddings_generated": 48,
      "processing_time_seconds": 28,
      "indexed_at": "2026-02-05T10:30:00Z"
    }
    ```

    **Design Notes:**

    - Async processing (return immediately, poll for status)
    - Support multiple document formats
    - Extract rich metadata for filtering
    - Flexible chunking strategies
    - Provide processing feedback for debugging

    ---

    ### 3. Conversation Management API

    **Create Conversation:**
    ```http
    POST /api/v1/conversations

    {
      "user_id": "user_123",
      "title": "Architecture Questions",
      "metadata": {
        "project": "backend-redesign",
        "team": "platform"
      }
    }

    Response:
    {
      "conversation_id": "conv_123",
      "created_at": "2026-02-05T10:00:00Z"
    }
    ```

    **Get Conversation History:**
    ```http
    GET /api/v1/conversations/conv_123

    {
      "conversation_id": "conv_123",
      "title": "Architecture Questions",
      "messages": [
        {
          "message_id": "msg_1",
          "role": "user",
          "content": "What are microservices?",
          "timestamp": "2026-02-05T10:05:00Z"
        },
        {
          "message_id": "msg_2",
          "role": "assistant",
          "content": "Microservices are...",
          "sources": [...],
          "timestamp": "2026-02-05T10:05:03Z"
        },
        // ... more messages
      ],
      "metadata": {
        "total_messages": 8,
        "total_tokens": 12500,
        "total_cost_usd": 0.025
      }
    }
    ```

    ---

    ## Database Schema

    ### Documents (PostgreSQL)

    ```sql
    CREATE TABLE documents (
      document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      title VARCHAR(500) NOT NULL,
      author VARCHAR(200),
      document_type VARCHAR(50), -- pdf, docx, txt, html, md
      file_path TEXT NOT NULL, -- S3 key
      file_size_bytes BIGINT,
      content_hash VARCHAR(64), -- SHA-256 for deduplication
      source_url TEXT,
      language VARCHAR(10) DEFAULT 'en',
      tags TEXT[], -- Array of tags
      access_level VARCHAR(50) DEFAULT 'internal',
      uploaded_by UUID,
      uploaded_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP,
      indexed_at TIMESTAMP,
      status VARCHAR(20) DEFAULT 'processing', -- processing, completed, failed
      chunk_count INT DEFAULT 0,
      metadata JSONB, -- Flexible metadata
      CONSTRAINT unique_content UNIQUE (content_hash)
    );

    CREATE INDEX idx_documents_type ON documents(document_type);
    CREATE INDEX idx_documents_tags ON documents USING GIN(tags);
    CREATE INDEX idx_documents_uploaded_at ON documents(uploaded_at DESC);
    CREATE INDEX idx_documents_status ON documents(status);
    ```

    ---

    ### Document Chunks (PostgreSQL)

    ```sql
    CREATE TABLE document_chunks (
      chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
      chunk_index INT NOT NULL, -- Position in document
      content TEXT NOT NULL, -- Actual chunk text
      content_hash VARCHAR(64), -- For deduplication
      token_count INT,
      start_char_index INT,
      end_char_index INT,
      page_number INT, -- For PDFs
      section_title VARCHAR(500), -- Extracted header
      embedding_id VARCHAR(100), -- Reference to vector DB
      created_at TIMESTAMP DEFAULT NOW(),
      metadata JSONB, -- Chunk-specific metadata
      CONSTRAINT unique_chunk UNIQUE (document_id, chunk_index)
    );

    CREATE INDEX idx_chunks_document ON document_chunks(document_id);
    CREATE INDEX idx_chunks_content_hash ON document_chunks(content_hash);
    ```

    ---

    ### Conversations (DynamoDB)

    **Design rationale:** DynamoDB for high availability and simple access patterns

    ```
    Table: conversations
    Partition Key: conversation_id (String)
    Sort Key: message_timestamp (Number, Unix timestamp)

    Attributes:
    {
      "conversation_id": "conv_123",
      "user_id": "user_123",
      "title": "Architecture Questions",
      "message_id": "msg_1",
      "message_timestamp": 1675608000,
      "role": "user" | "assistant",
      "content": "What are microservices?",
      "query_id": "query_xyz789",  // For assistant messages
      "sources": [
        {
          "citation_id": "[1]",
          "document_id": "doc_789",
          "chunk_id": "chunk_123",
          "relevance_score": 0.92
        }
      ],
      "metadata": {
        "tokens_used": 2450,
        "cost_usd": 0.0049,
        "model": "gpt-4",
        "latency_ms": 2410
      },
      "created_at": "2026-02-05T10:05:00Z"
    }

    GSI: user_id-message_timestamp-index (for user's conversation history)
    ```

    ---

    ### Vector Embeddings (Pinecone/Weaviate)

    **Pinecone Index Schema:**

    ```python
    # Index configuration
    {
      "name": "rag-embeddings",
      "dimension": 1536,  # OpenAI ada-002 or 768 for sentence-transformers
      "metric": "cosine", # cosine, euclidean, dotproduct
      "pod_type": "p1.x1",
      "replicas": 2,
      "shards": 4
    }

    # Vector record structure
    {
      "id": "chunk_123",  # chunk_id from PostgreSQL
      "values": [0.023, -0.45, ...],  # 1536-dim embedding
      "metadata": {
        "document_id": "doc_789",
        "document_title": "Microservices Design Patterns",
        "document_type": "pdf",
        "chunk_index": 12,
        "page_number": 34,
        "section_title": "Scalability Patterns",
        "tags": ["architecture", "backend"],
        "uploaded_at": "2026-02-05T10:00:00Z",
        "content_preview": "Microservices allow independent scaling..."  # First 200 chars
      }
    }
    ```

    **Weaviate Schema Alternative:**

    ```graphql
    {
      "class": "DocumentChunk",
      "vectorizer": "none",  # We provide embeddings
      "properties": [
        {"name": "chunkId", "dataType": ["string"]},
        {"name": "documentId", "dataType": ["string"]},
        {"name": "documentTitle", "dataType": ["string"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "chunkIndex", "dataType": ["int"]},
        {"name": "pageNumber", "dataType": ["int"]},
        {"name": "tags", "dataType": ["string[]"]},
        {"name": "uploadedAt", "dataType": ["date"]}
      ]
    }
    ```

    ---

    ### Elasticsearch Index (Keyword Search)

    ```json
    {
      "settings": {
        "analysis": {
          "analyzer": {
            "custom_analyzer": {
              "type": "custom",
              "tokenizer": "standard",
              "filter": ["lowercase", "stop", "snowball"]
            }
          }
        },
        "index": {
          "number_of_shards": 5,
          "number_of_replicas": 1
        }
      },
      "mappings": {
        "properties": {
          "chunk_id": {"type": "keyword"},
          "document_id": {"type": "keyword"},
          "document_title": {
            "type": "text",
            "analyzer": "custom_analyzer",
            "fields": {"keyword": {"type": "keyword"}}
          },
          "content": {
            "type": "text",
            "analyzer": "custom_analyzer"
          },
          "section_title": {"type": "text"},
          "tags": {"type": "keyword"},
          "document_type": {"type": "keyword"},
          "page_number": {"type": "integer"},
          "uploaded_at": {"type": "date"}
        }
      }
    }
    ```

=== "🔍 Step 3: Deep Dive"

    ## 1. Document Chunking Strategies

    **Challenge:** Documents need to be split into chunks that fit LLM context windows while maintaining semantic coherence.

    ### Chunking Strategies

    **1. Fixed-Size Chunking**

    Simple but effective for uniform content:

    ```python
    def fixed_size_chunking(text: str, chunk_size: int = 1000,
                            chunk_overlap: int = 200) -> List[str]:
        """
        Split text into fixed-size chunks with overlap

        Args:
            text: Input document text
            chunk_size: Number of tokens per chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        tokens = tokenizer.encode(text)

        chunks = []
        stride = chunk_size - chunk_overlap

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            if i + chunk_size >= len(tokens):
                break

        return chunks
    ```

    **Pros:** Simple, predictable chunk sizes
    **Cons:** May split mid-sentence, no semantic awareness

    ---

    **2. Recursive Character Text Splitting (LangChain)**

    Hierarchical splitting that respects document structure:

    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def recursive_chunking(text: str, chunk_size: int = 1000,
                          chunk_overlap: int = 200) -> List[str]:
        """
        Recursively split text by separators (paragraph, sentence, word)

        Tries to split by:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Sentences (periods)
        4. Words (spaces)
        5. Characters (fallback)

        Returns clean chunks that respect document structure
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        return chunks
    ```

    **Pros:** Respects natural boundaries, maintains context
    **Cons:** Variable chunk sizes, may still split important content

    ---

    **3. Semantic Chunking**

    Split based on semantic similarity between sentences:

    ```python
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from typing import List

    class SemanticChunker:
        """
        Chunk text based on semantic similarity between sentences
        Groups consecutive sentences with high similarity
        """

        def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                     similarity_threshold: float = 0.7):
            self.model = SentenceTransformer(model_name)
            self.threshold = similarity_threshold

        def chunk(self, text: str, max_chunk_size: int = 1000) -> List[str]:
            """
            Split text into semantically coherent chunks

            Args:
                text: Input document
                max_chunk_size: Maximum chunk size in tokens

            Returns:
                List of semantic chunks
            """
            # Split into sentences
            sentences = self._split_sentences(text)

            # Generate embeddings for all sentences
            embeddings = self.model.encode(sentences)

            # Calculate similarity between consecutive sentences
            similarities = self._cosine_similarity(embeddings[:-1], embeddings[1:])

            # Group sentences based on similarity
            chunks = []
            current_chunk = [sentences[0]]
            current_size = len(sentences[0].split())

            for i, similarity in enumerate(similarities):
                next_sentence = sentences[i + 1]
                next_size = len(next_sentence.split())

                # Start new chunk if similarity drops or size exceeds limit
                if similarity < self.threshold or current_size + next_size > max_chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [next_sentence]
                    current_size = next_size
                else:
                    current_chunk.append(next_sentence)
                    current_size += next_size

            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        def _split_sentences(self, text: str) -> List[str]:
            """Split text into sentences using NLTK or spaCy"""
            import nltk
            nltk.download('punkt', quiet=True)
            return nltk.sent_tokenize(text)

        def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Calculate cosine similarity between consecutive embeddings"""
            return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    ```

    **Pros:** Maintains semantic coherence, better context preservation
    **Cons:** Computationally expensive, variable chunk sizes

    ---

    **4. Structure-Aware Chunking**

    Extract and preserve document structure (headers, sections, lists):

    ```python
    from typing import List, Dict
    import re

    class StructureAwareChunker:
        """
        Chunk documents while preserving structure (headers, sections)
        Useful for technical docs, wikis, manuals
        """

        def chunk_markdown(self, text: str, max_chunk_size: int = 1000) -> List[Dict]:
            """
            Chunk markdown while preserving section structure

            Returns chunks with metadata about document structure
            """
            chunks = []
            current_section = {"headers": [], "content": []}

            lines = text.split('\n')
            current_size = 0

            for line in lines:
                # Detect markdown headers
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2)

                    # Save previous chunk if it exists
                    if current_section["content"]:
                        chunks.append(self._create_chunk(current_section))
                        current_section = {"headers": [], "content": []}
                        current_size = 0

                    # Update header hierarchy
                    current_section["headers"] = current_section["headers"][:level-1] + [title]

                else:
                    # Add content line
                    line_size = len(line.split())
                    if current_size + line_size > max_chunk_size:
                        # Save chunk and start new one
                        chunks.append(self._create_chunk(current_section))
                        current_section = {
                            "headers": current_section["headers"],
                            "content": [line]
                        }
                        current_size = line_size
                    else:
                        current_section["content"].append(line)
                        current_size += line_size

            # Add final chunk
            if current_section["content"]:
                chunks.append(self._create_chunk(current_section))

            return chunks

        def _create_chunk(self, section: Dict) -> Dict:
            """Create chunk with structural metadata"""
            return {
                "content": "\n".join(section["content"]),
                "section_title": " > ".join(section["headers"]) if section["headers"] else None,
                "headers": section["headers"],
                "metadata": {
                    "depth": len(section["headers"])
                }
            }
    ```

    **Pros:** Preserves document structure, better for technical docs
    **Cons:** Format-specific, may need parsers for different formats

    ---

    ### Chunking Best Practices

    | Aspect | Recommendation |
    |--------|---------------|
    | **Chunk Size** | 500-1500 tokens (balance between context and specificity) |
    | **Overlap** | 10-20% of chunk size (maintains context continuity) |
    | **Strategy** | Recursive for general content, semantic for narrative, structure-aware for technical docs |
    | **Metadata** | Include section titles, page numbers, document hierarchy |
    | **Deduplication** | Hash chunks to avoid indexing duplicates |

    ---

    ## 2. Embedding Generation

    **Challenge:** Convert text chunks into dense vector representations that capture semantic meaning.

    ### Embedding Models Comparison

    | Model | Dimension | Max Tokens | Speed | Cost | Best For |
    |-------|-----------|------------|-------|------|----------|
    | **OpenAI ada-002** | 1536 | 8191 | 0.1s | $0.0001/1K tokens | General purpose, high quality |
    | **Sentence-Transformers (all-MiniLM-L6-v2)** | 384 | 256 | 0.05s | Free (self-hosted) | Fast, lightweight, cost-effective |
    | **Sentence-Transformers (all-mpnet-base-v2)** | 768 | 384 | 0.08s | Free (self-hosted) | Better quality than MiniLM |
    | **Cohere Embed v3** | 1024 | 512 | 0.12s | $0.0001/1K tokens | Multilingual, customizable |
    | **Voyage AI** | 1024 | 16000 | 0.15s | $0.0001/1K tokens | Long context, high quality |

    ---

    ### Embedding Service Implementation

    ```python
    from typing import List
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import openai
    from functools import lru_cache

    class EmbeddingService:
        """
        Unified embedding service supporting multiple providers
        """

        def __init__(self, provider: str = "sentence-transformers",
                     model_name: str = "all-mpnet-base-v2",
                     cache_size: int = 10000):
            self.provider = provider
            self.model_name = model_name

            if provider == "sentence-transformers":
                self.model = SentenceTransformer(model_name)
            elif provider == "openai":
                openai.api_key = os.getenv("OPENAI_API_KEY")

        @lru_cache(maxsize=10000)
        def embed_query(self, text: str) -> np.ndarray:
            """
            Generate embedding for a single query (with caching)

            Args:
                text: Query text

            Returns:
                Embedding vector
            """
            return self._embed([text])[0]

        def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
            """
            Generate embeddings for multiple documents (batched)

            Args:
                texts: List of document chunks
                batch_size: Batch size for processing

            Returns:
                Array of embeddings (N, embedding_dim)
            """
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embed(batch)
                embeddings.extend(batch_embeddings)

            return np.array(embeddings)

        def _embed(self, texts: List[str]) -> List[np.ndarray]:
            """Internal embedding generation"""
            if self.provider == "sentence-transformers":
                return self.model.encode(texts, convert_to_numpy=True)

            elif self.provider == "openai":
                response = openai.Embedding.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )
                return [item['embedding'] for item in response['data']]

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        def similarity(self, query_embedding: np.ndarray,
                      document_embeddings: np.ndarray) -> np.ndarray:
            """
            Calculate cosine similarity between query and documents

            Args:
                query_embedding: Query vector (embedding_dim,)
                document_embeddings: Document vectors (N, embedding_dim)

            Returns:
                Similarity scores (N,)
            """
            # Cosine similarity = dot product of normalized vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)

            similarities = np.dot(doc_norms, query_norm)
            return similarities
    ```

    ---

    ### Embedding Optimization

    **1. Dimensionality Reduction**

    Reduce embedding size for faster search:

    ```python
    from sklearn.decomposition import PCA

    class EmbeddingCompressor:
        """
        Compress embeddings using PCA
        Trade slight quality loss for 2-4x speed improvement
        """

        def __init__(self, original_dim: int = 1536, target_dim: int = 768):
            self.pca = PCA(n_components=target_dim)
            self.fitted = False

        def fit(self, embeddings: np.ndarray):
            """Fit PCA on training embeddings"""
            self.pca.fit(embeddings)
            self.fitted = True

        def compress(self, embeddings: np.ndarray) -> np.ndarray:
            """Compress embeddings to lower dimension"""
            if not self.fitted:
                raise ValueError("Must call fit() first")
            return self.pca.transform(embeddings)

        def explained_variance(self) -> float:
            """Get percentage of variance retained"""
            return np.sum(self.pca.explained_variance_ratio_)
    ```

    ---

    **2. Matryoshka Embeddings**

    Modern approach: embeddings where smaller dimensions work independently:

    ```python
    # Using Matryoshka-capable models (Nomic, Cohere)
    # Can truncate dimensions without quality loss

    # Full embedding: 1536 dimensions
    full_embedding = embed("text")

    # Truncate to 768 dimensions (2x faster search)
    # Quality degradation: < 5%
    truncated_embedding = full_embedding[:768]

    # Truncate to 384 dimensions (4x faster search)
    # Quality degradation: < 10%
    truncated_embedding = full_embedding[:384]
    ```

    ---

    ## 3. Retrieval Strategies

    **Challenge:** Find the most relevant chunks from millions of candidates in < 50ms.

    ### Retrieval Pipeline

    ```
    Query → [Candidate Generation] → [Reranking] → [Filtering] → Final Results
            (Fast, 1000s)           (Accurate, 100s)  (Business)   (Top-10)
    ```

    ---

    ### A. Semantic Search (Dense Retrieval)

    Vector similarity using ANN (Approximate Nearest Neighbor):

    ```python
    import pinecone
    from typing import List, Dict

    class SemanticRetriever:
        """
        Semantic search using vector database
        """

        def __init__(self, index_name: str, api_key: str):
            pinecone.init(api_key=api_key, environment="us-west1-gcp")
            self.index = pinecone.Index(index_name)

        def search(self, query_embedding: List[float], top_k: int = 100,
                  filters: Dict = None) -> List[Dict]:
            """
            Semantic search with optional metadata filters

            Args:
                query_embedding: Query vector
                top_k: Number of results
                filters: Metadata filters (e.g., {"document_type": "pdf"})

            Returns:
                List of matches with scores
            """
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filters,  # e.g., {"tags": {"$in": ["architecture"]}}
                include_metadata=True
            )

            return [
                {
                    "chunk_id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
    ```

    **Pros:** Captures semantic intent, works for paraphrased queries
    **Cons:** May miss exact keyword matches, requires expensive embeddings

    ---

    ### B. Keyword Search (Sparse Retrieval)

    BM25 algorithm for lexical matching:

    ```python
    from elasticsearch import Elasticsearch
    from typing import List, Dict

    class KeywordRetriever:
        """
        Keyword search using Elasticsearch BM25
        """

        def __init__(self, es_host: str, index_name: str):
            self.es = Elasticsearch([es_host])
            self.index = index_name

        def search(self, query: str, top_k: int = 100,
                  filters: Dict = None) -> List[Dict]:
            """
            BM25 keyword search

            Args:
                query: Query string
                top_k: Number of results
                filters: Filters (e.g., {"document_type": "pdf"})

            Returns:
                List of matches with BM25 scores
            """
            # Build Elasticsearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "section_title^1.5", "document_title"],
                                    "type": "best_fields",
                                    "operator": "or"
                                }
                            }
                        ],
                        "filter": self._build_filters(filters)
                    }
                },
                "size": top_k,
                "_source": ["chunk_id", "document_id", "content", "metadata"]
            }

            response = self.es.search(index=self.index, body=es_query)

            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"],
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"].get("metadata", {})
                }
                for hit in response["hits"]["hits"]
            ]

        def _build_filters(self, filters: Dict) -> List[Dict]:
            """Convert filters to Elasticsearch filter clauses"""
            if not filters:
                return []

            filter_clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})

            return filter_clauses
    ```

    **Pros:** Exact keyword matching, no embedding required, fast
    **Cons:** No semantic understanding, sensitive to vocabulary mismatch

    ---

    ### C. Hybrid Retrieval (Best of Both Worlds)

    Combine semantic and keyword search using Reciprocal Rank Fusion:

    ```python
    from typing import List, Dict
    import numpy as np

    class HybridRetriever:
        """
        Hybrid search combining semantic + keyword retrieval
        Uses Reciprocal Rank Fusion (RRF) for score combination
        """

        def __init__(self, semantic_retriever: SemanticRetriever,
                     keyword_retriever: KeywordRetriever):
            self.semantic = semantic_retriever
            self.keyword = keyword_retriever

        def search(self, query: str, query_embedding: List[float],
                  top_k: int = 100, alpha: float = 0.7,
                  filters: Dict = None) -> List[Dict]:
            """
            Hybrid search with configurable weight

            Args:
                query: Query string (for keyword search)
                query_embedding: Query vector (for semantic search)
                top_k: Number of final results
                alpha: Weight for semantic vs keyword (0-1)
                       1.0 = pure semantic, 0.0 = pure keyword
                filters: Metadata filters

            Returns:
                Fused results sorted by combined score
            """
            # Get candidates from both retrievers
            semantic_results = self.semantic.search(
                query_embedding, top_k=top_k*2, filters=filters
            )
            keyword_results = self.keyword.search(
                query, top_k=top_k*2, filters=filters
            )

            # Reciprocal Rank Fusion
            fused_scores = {}

            # Add semantic scores
            for rank, result in enumerate(semantic_results):
                chunk_id = result["chunk_id"]
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (60 + rank + 1)  # k=60 is standard
                fused_scores[chunk_id] = {
                    "semantic_score": result["score"],
                    "semantic_rrf": rrf_score * alpha,
                    "keyword_rrf": 0,
                    "metadata": result.get("metadata", {})
                }

            # Add keyword scores
            for rank, result in enumerate(keyword_results):
                chunk_id = result["chunk_id"]
                rrf_score = 1.0 / (60 + rank + 1)

                if chunk_id in fused_scores:
                    fused_scores[chunk_id]["keyword_rrf"] = rrf_score * (1 - alpha)
                else:
                    fused_scores[chunk_id] = {
                        "semantic_rrf": 0,
                        "keyword_rrf": rrf_score * (1 - alpha),
                        "metadata": result.get("metadata", {})
                    }

            # Calculate final scores and sort
            results = []
            for chunk_id, scores in fused_scores.items():
                final_score = scores["semantic_rrf"] + scores["keyword_rrf"]
                results.append({
                    "chunk_id": chunk_id,
                    "score": final_score,
                    "semantic_score": scores.get("semantic_score"),
                    "metadata": scores["metadata"]
                })

            # Sort by final score and return top-k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
    ```

    **Why RRF instead of score normalization?**

    - Rank-based fusion is more robust than score-based
    - Semantic and keyword scores have different scales (cosine 0-1 vs BM25 0-∞)
    - RRF treats both retrievers fairly regardless of score ranges

    ---

    ### D. Reranking (Cross-Encoder)

    Rerank top candidates using a more expensive but accurate model:

    ```python
    from sentence_transformers import CrossEncoder
    from typing import List, Dict

    class Reranker:
        """
        Rerank retrieved chunks using cross-encoder model
        Much slower but more accurate than bi-encoder retrieval
        """

        def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            self.model = CrossEncoder(model_name)

        def rerank(self, query: str, candidates: List[Dict],
                   top_k: int = 10) -> List[Dict]:
            """
            Rerank candidates using cross-encoder

            Args:
                query: Original query
                candidates: Retrieved chunks with content
                top_k: Number of results to return

            Returns:
                Reranked results
            """
            # Prepare pairs for cross-encoder
            pairs = [(query, candidate["content"]) for candidate in candidates]

            # Score all pairs
            scores = self.model.predict(pairs)

            # Add scores to candidates
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            # Sort by rerank score
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            return candidates[:top_k]
    ```

    **When to use reranking:**

    - When accuracy is critical (legal, medical, financial)
    - After hybrid retrieval to refine top-100 to top-10
    - When you can afford extra latency (50-200ms)

    ---

    ### E. Maximal Marginal Relevance (MMR)

    Diversify results to avoid redundancy:

    ```python
    import numpy as np
    from typing import List, Dict

    class MMRSelector:
        """
        Select diverse results using Maximal Marginal Relevance
        Balances relevance and diversity
        """

        def select(self, query_embedding: np.ndarray,
                  candidate_embeddings: np.ndarray,
                  candidates: List[Dict],
                  top_k: int = 10,
                  lambda_param: float = 0.5) -> List[Dict]:
            """
            Select top-k diverse results using MMR

            Args:
                query_embedding: Query vector
                candidate_embeddings: Candidate vectors (N, dim)
                candidates: Candidate metadata
                top_k: Number of results
                lambda_param: Relevance vs diversity (0-1)
                             1.0 = pure relevance, 0.0 = pure diversity

            Returns:
                Selected candidates (diverse + relevant)
            """
            # Calculate query similarities
            query_sims = self._cosine_similarity(
                query_embedding.reshape(1, -1),
                candidate_embeddings
            )[0]

            selected_indices = []
            remaining_indices = list(range(len(candidates)))

            for _ in range(min(top_k, len(candidates))):
                mmr_scores = []

                for i in remaining_indices:
                    # Relevance to query
                    relevance = query_sims[i]

                    # Max similarity to already selected
                    if selected_indices:
                        selected_embeddings = candidate_embeddings[selected_indices]
                        max_sim = np.max(self._cosine_similarity(
                            candidate_embeddings[i:i+1],
                            selected_embeddings
                        ))
                    else:
                        max_sim = 0

                    # MMR formula
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append((i, mmr))

                # Select best MMR score
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            return [candidates[i] for i in selected_indices]

        def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Calculate cosine similarity"""
            a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
            return np.dot(a_norm, b_norm.T)
    ```

    ---

    ## 4. Prompt Engineering for RAG

    **Challenge:** Construct prompts that guide the LLM to generate accurate, cited responses.

    ### RAG Prompt Template

    ```python
    from typing import List, Dict

    class RAGPromptBuilder:
        """
        Build prompts for RAG with retrieved context
        """

        def build_prompt(self, query: str, contexts: List[Dict],
                        conversation_history: List[Dict] = None,
                        system_instructions: str = None) -> str:
            """
            Build prompt with retrieved contexts and conversation history

            Args:
                query: User question
                contexts: Retrieved chunks with content and metadata
                conversation_history: Previous messages
                system_instructions: Custom system prompt

            Returns:
                Complete prompt string
            """
            # Default system instructions
            if not system_instructions:
                system_instructions = """You are a helpful AI assistant that answers questions based on provided context.

CRITICAL RULES:
1. ONLY use information from the provided context documents
2. Cite sources using [N] notation (e.g., "According to the design doc [1]...")
3. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer that question"
4. DO NOT make up information or use knowledge outside the provided context
5. If multiple sources provide conflicting information, mention both and cite appropriately
6. Be concise but complete in your answers
"""

            # Build context section
            context_section = self._build_context_section(contexts)

            # Build conversation history section
            history_section = ""
            if conversation_history:
                history_section = self._build_history_section(conversation_history)

            # Assemble full prompt
            prompt = f"""{system_instructions}

{context_section}

{history_section}

User Question: {query}

Answer (with citations):"""
            return prompt

        def _build_context_section(self, contexts: List[Dict]) -> str:
            """Format context documents with citations"""
            context_lines = ["Context Documents:\n"]

            for i, ctx in enumerate(contexts, 1):
                citation_id = f"[{i}]"
                document_title = ctx.get("document_title", "Unknown")
                page = ctx.get("page_number", "N/A")
                content = ctx.get("content", "")

                context_lines.append(
                    f"{citation_id} (Source: {document_title}, Page {page})\n{content}\n"
                )

            return "\n".join(context_lines)

        def _build_history_section(self, history: List[Dict]) -> str:
            """Format conversation history"""
            if not history:
                return ""

            history_lines = ["Previous Conversation:\n"]

            for msg in history[-5:]:  # Only last 5 messages
                role = msg["role"]
                content = msg["content"]
                history_lines.append(f"{role.capitalize()}: {content}\n")

            return "\n".join(history_lines)
    ```

    ---

    ### Advanced Prompt Techniques

    **1. Query Decomposition**

    Break complex queries into simpler sub-queries:

    ```python
    class QueryDecomposer:
        """
        Decompose complex queries into sub-queries
        """

        def __init__(self, llm_client):
            self.llm = llm_client

        def decompose(self, query: str) -> List[str]:
            """
            Break down complex query into simple sub-queries

            Example:
                "Compare microservices and monolithic architectures"
                →
                ["What are the benefits of microservices?",
                 "What are the benefits of monolithic architectures?",
                 "What are the trade-offs between them?"]
            """
            decomposition_prompt = f"""Break down this complex question into 2-4 simpler sub-questions that can be answered independently.

Question: {query}

Sub-questions (one per line):"""

            response = self.llm.complete(decomposition_prompt, max_tokens=200)
            sub_queries = [q.strip() for q in response.split('\n') if q.strip()]

            return sub_queries
    ```

    ---

    **2. Self-Reflection for Quality**

    Have LLM verify its answer against context:

    ```python
    class AnswerValidator:
        """
        Validate generated answers for hallucinations
        """

        def validate(self, query: str, answer: str, contexts: List[str]) -> Dict:
            """
            Check if answer is grounded in provided contexts

            Returns:
                {
                    "is_valid": bool,
                    "issues": List[str],
                    "confidence": float
                }
            """
            validation_prompt = f"""Given the following context and answer, determine if the answer is fully supported by the context.

Context:
{self._format_contexts(contexts)}

Question: {query}
Answer: {answer}

Is the answer fully supported by the context? Check for:
1. Factual accuracy
2. No hallucinations
3. Proper citations
4. No external knowledge used

Response format:
VALID: yes/no
ISSUES: [list any issues]
CONFIDENCE: [0.0-1.0]"""

            response = self.llm.complete(validation_prompt)
            return self._parse_validation_response(response)
    ```

    ---

    ## 5. Real-time Document Updates & Incremental Indexing

    **Challenge:** Add new documents and update existing ones without full reindexing.

    ### Incremental Indexing Pipeline

    ```python
    from typing import List, Dict
    import hashlib

    class IncrementalIndexer:
        """
        Handle document updates with incremental indexing
        """

        def __init__(self, doc_store, vector_db, search_index, metadata_db):
            self.doc_store = doc_store
            self.vector_db = vector_db
            self.search_index = search_index
            self.metadata_db = metadata_db

        def update_document(self, document_id: str, new_content: str,
                           metadata: Dict = None):
            """
            Update existing document with incremental reindexing

            Steps:
            1. Check if content changed (hash comparison)
            2. Delete old chunks from vector DB and search index
            3. Generate new chunks and embeddings
            4. Update indexes
            5. Update metadata
            """
            # Get existing document
            existing_doc = self.metadata_db.get_document(document_id)
            if not existing_doc:
                raise ValueError(f"Document {document_id} not found")

            # Check if content changed
            new_hash = self._compute_hash(new_content)
            if new_hash == existing_doc["content_hash"]:
                print("Content unchanged, skipping reindexing")
                if metadata:
                    self.metadata_db.update_metadata(document_id, metadata)
                return

            # Delete old chunks
            old_chunk_ids = self.metadata_db.get_chunk_ids(document_id)
            self._delete_chunks(old_chunk_ids)

            # Process new content
            chunks = self._chunk_document(new_content)
            embeddings = self._generate_embeddings(chunks)

            # Index new chunks
            chunk_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"

                # Store in vector DB
                self.vector_db.upsert(
                    id=chunk_id,
                    values=embedding,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk["content"][:200]  # Preview
                    }
                )

                # Store in search index
                self.search_index.index(
                    id=chunk_id,
                    document={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "content": chunk["content"]
                    }
                )

                chunk_records.append({
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "chunk_index": i
                })

            # Update metadata DB
            self.metadata_db.update_document(
                document_id=document_id,
                content_hash=new_hash,
                chunk_count=len(chunks),
                updated_at=datetime.now(),
                metadata=metadata
            )
            self.metadata_db.insert_chunks(document_id, chunk_records)

        def _delete_chunks(self, chunk_ids: List[str]):
            """Delete chunks from all indexes"""
            # Delete from vector DB
            self.vector_db.delete(ids=chunk_ids)

            # Delete from search index
            for chunk_id in chunk_ids:
                self.search_index.delete(chunk_id)

            # Delete from metadata DB
            self.metadata_db.delete_chunks(chunk_ids)

        def _compute_hash(self, content: str) -> str:
            """Compute SHA-256 hash for deduplication"""
            return hashlib.sha256(content.encode()).hexdigest()

        def _chunk_document(self, content: str) -> List[Dict]:
            """Chunk document using configured strategy"""
            # Implementation depends on chunking strategy
            pass

        def _generate_embeddings(self, chunks: List[Dict]) -> List[np.ndarray]:
            """Generate embeddings for chunks"""
            # Implementation depends on embedding model
            pass
    ```

    ---

    ### Change Data Capture (CDC) for Real-time Updates

    Use message queue to propagate updates:

    ```python
    import json
    from kafka import KafkaConsumer, KafkaProducer

    class DocumentUpdateProcessor:
        """
        Process document updates from Kafka stream
        """

        def __init__(self, indexer: IncrementalIndexer):
            self.indexer = indexer
            self.consumer = KafkaConsumer(
                'document-updates',
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )

        def process_updates(self):
            """
            Continuously process document updates

            Event format:
            {
                "event_type": "create" | "update" | "delete",
                "document_id": "doc_123",
                "content": "...",  # For create/update
                "metadata": {...}
            }
            """
            for message in self.consumer:
                event = message.value

                try:
                    if event["event_type"] == "create":
                        self._handle_create(event)
                    elif event["event_type"] == "update":
                        self._handle_update(event)
                    elif event["event_type"] == "delete":
                        self._handle_delete(event)

                except Exception as e:
                    print(f"Error processing event: {e}")
                    # Send to dead letter queue
                    self._send_to_dlq(event, str(e))

        def _handle_create(self, event):
            """Handle new document creation"""
            self.indexer.index_document(
                document_id=event["document_id"],
                content=event["content"],
                metadata=event.get("metadata", {})
            )

        def _handle_update(self, event):
            """Handle document update"""
            self.indexer.update_document(
                document_id=event["document_id"],
                new_content=event["content"],
                metadata=event.get("metadata")
            )

        def _handle_delete(self, event):
            """Handle document deletion"""
            self.indexer.delete_document(event["document_id"])
    ```

    ---

    ## 6. LangChain Integration Example

    **End-to-end RAG system using LangChain:**

    ```python
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate

    class LangChainRAGSystem:
        """
        Complete RAG system using LangChain
        """

        def __init__(self, pinecone_index_name: str):
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

            # Initialize vector store
            self.vector_store = Pinecone.from_existing_index(
                index_name=pinecone_index_name,
                embedding=self.embeddings
            )

            # Initialize LLM
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                max_tokens=500
            )

            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Custom prompt template
            prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer based on the context, say so. Always cite your sources using [N] notation.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer with citations:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )

            # Create retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",  # Use MMR for diversity
                    search_kwargs={
                        "k": 10,
                        "fetch_k": 50,
                        "lambda_mult": 0.5
                    }
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )

        def query(self, question: str) -> Dict:
            """
            Query the RAG system

            Returns:
                {
                    "answer": str,
                    "source_documents": List[Document]
                }
            """
            result = self.chain({"question": question})

            return {
                "answer": result["answer"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }

        def clear_history(self):
            """Clear conversation memory"""
            self.memory.clear()
    ```

    ---

=== "⚡ Step 4: Scalability"

    ## Scaling Challenges

    | Challenge | Solution |
    |-----------|----------|
    | **Vector search latency** | Sharding, HNSW index tuning, GPU acceleration |
    | **Embedding generation bottleneck** | Batch processing, multiple embedding service replicas |
    | **LLM rate limits** | Request queuing, multiple API keys, caching |
    | **Large document processing** | Distributed chunking with Spark, async processing |
    | **High query throughput** | Read replicas, multi-level caching, CDN for responses |

    ---

    ## Vector Database Sharding

    **Horizontal partitioning for billion-scale embeddings:**

    ```python
    from typing import List
    import hashlib

    class ShardedVectorDB:
        """
        Shard vector database across multiple indexes
        """

        def __init__(self, shard_configs: List[Dict]):
            """
            shard_configs: [
                {"shard_id": 0, "index_name": "rag-shard-0", ...},
                {"shard_id": 1, "index_name": "rag-shard-1", ...}
            ]
            """
            self.shards = []
            for config in shard_configs:
                shard = pinecone.Index(config["index_name"])
                self.shards.append({
                    "shard_id": config["shard_id"],
                    "index": shard
                })

        def _get_shard(self, document_id: str) -> int:
            """Consistent hashing to determine shard"""
            hash_value = int(hashlib.md5(document_id.encode()).hexdigest(), 16)
            return hash_value % len(self.shards)

        def upsert(self, chunk_id: str, embedding: List[float], metadata: Dict):
            """Insert embedding into appropriate shard"""
            document_id = metadata.get("document_id")
            shard_idx = self._get_shard(document_id)

            self.shards[shard_idx]["index"].upsert(
                vectors=[(chunk_id, embedding, metadata)]
            )

        def search(self, query_embedding: List[float], top_k: int = 10,
                  filters: Dict = None) -> List[Dict]:
            """
            Search across all shards and merge results

            Strategy: Query all shards in parallel, merge top results
            """
            from concurrent.futures import ThreadPoolExecutor

            def search_shard(shard):
                results = shard["index"].query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filters,
                    include_metadata=True
                )
                return results.matches

            # Parallel search across shards
            with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
                shard_results = executor.map(
                    search_shard,
                    self.shards
                )

            # Merge and sort results from all shards
            all_results = []
            for results in shard_results:
                all_results.extend(results)

            # Sort by score and return top-k
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]
    ```

    ---

    ## Multi-Level Caching Strategy

    **Reduce latency and costs with intelligent caching:**

    ```python
    import redis
    import json
    from functools import wraps
    import hashlib

    class RAGCacheManager:
        """
        Multi-level cache for RAG system
        L1: Query embeddings (fast lookup)
        L2: Query results (entire response)
        L3: Retrieved chunks (reduce vector search)
        """

        def __init__(self, redis_client: redis.Redis):
            self.redis = redis_client

        def cache_query_result(self, ttl: int = 3600):
            """
            Decorator to cache query results

            Cache key: hash(query + filters)
            TTL: 1 hour (configurable)
            """
            def decorator(func):
                @wraps(func)
                def wrapper(query: str, filters: Dict = None, *args, **kwargs):
                    # Generate cache key
                    cache_key = self._generate_query_key(query, filters)

                    # Check cache
                    cached = self.redis.get(cache_key)
                    if cached:
                        return json.loads(cached)

                    # Execute query
                    result = func(query, filters, *args, **kwargs)

                    # Store in cache
                    self.redis.setex(
                        cache_key,
                        ttl,
                        json.dumps(result)
                    )

                    return result
                return wrapper
            return decorator

        def cache_embedding(self, query: str, embedding: List[float], ttl: int = 86400):
            """
            Cache query embedding (24-hour TTL)
            Saves embedding API calls
            """
            key = f"embedding:{self._hash(query)}"
            self.redis.setex(key, ttl, json.dumps(embedding))

        def get_cached_embedding(self, query: str) -> List[float]:
            """Retrieve cached embedding"""
            key = f"embedding:{self._hash(query)}"
            cached = self.redis.get(key)
            return json.loads(cached) if cached else None

        def _generate_query_key(self, query: str, filters: Dict) -> str:
            """Generate cache key from query + filters"""
            filter_str = json.dumps(filters, sort_keys=True) if filters else ""
            combined = f"{query}:{filter_str}"
            return f"query:{self._hash(combined)}"

        def _hash(self, text: str) -> str:
            """Generate MD5 hash"""
            return hashlib.md5(text.encode()).hexdigest()
    ```

    ---

    ## LLM Request Batching

    **Optimize LLM costs with batching:**

    ```python
    import asyncio
    from typing import List, Dict
    from collections import defaultdict

    class LLMBatcher:
        """
        Batch multiple LLM requests for cost optimization
        Wait for batch_size requests or max_wait_time
        """

        def __init__(self, llm_client, batch_size: int = 10,
                     max_wait_time: float = 0.5):
            self.llm = llm_client
            self.batch_size = batch_size
            self.max_wait_time = max_wait_time
            self.pending_requests = []
            self.request_futures = {}

        async def generate(self, prompt: str, request_id: str) -> str:
            """
            Add request to batch and wait for result

            Args:
                prompt: LLM prompt
                request_id: Unique request identifier

            Returns:
                Generated response
            """
            # Create future for this request
            future = asyncio.Future()
            self.request_futures[request_id] = future

            # Add to pending batch
            self.pending_requests.append({
                "request_id": request_id,
                "prompt": prompt
            })

            # Trigger batch processing if full
            if len(self.pending_requests) >= self.batch_size:
                asyncio.create_task(self._process_batch())
            else:
                # Start timer for partial batch
                asyncio.create_task(self._wait_and_process())

            # Wait for result
            return await future

        async def _wait_and_process(self):
            """Wait max_wait_time then process partial batch"""
            await asyncio.sleep(self.max_wait_time)
            if self.pending_requests:
                await self._process_batch()

        async def _process_batch(self):
            """Process batch of requests"""
            if not self.pending_requests:
                return

            # Extract batch
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]

            # Call LLM with batch
            prompts = [req["prompt"] for req in batch]
            responses = await self.llm.batch_generate(prompts)

            # Resolve futures
            for req, response in zip(batch, responses):
                future = self.request_futures.pop(req["request_id"])
                future.set_result(response)
    ```

    ---

    ## Cost Optimization

    **Monthly cost breakdown at 100K queries/day:**

    | Component | Usage | Cost |
    |-----------|-------|------|
    | **LLM API (GPT-4)** | 7.5B tokens/month × $0.03/1K | $225,000 |
    | **Embeddings (OpenAI)** | 3B tokens/month × $0.0001/1K | $300 |
    | **Vector DB (Pinecone)** | 500M vectors, 3M queries | $700 |
    | **Elasticsearch** | 3-node cluster, 500GB | $450 |
    | **PostgreSQL** | RDS r5.large | $180 |
    | **Redis Cache** | 100GB ElastiCache | $160 |
    | **EC2 (API servers)** | 10 × c5.2xlarge | $1,200 |
    | **S3 Storage** | 5TB | $115 |
    | **Data Transfer** | 10TB egress | $900 |
    | **Total** | | **$229,005/month** |

    **Cost optimization strategies:**

    1. **Use smaller LLM for simple queries** (GPT-3.5: 10x cheaper)
    2. **Cache responses aggressively** (20% cache hit = $45K savings)
    3. **Self-hosted embeddings** (Sentence-Transformers: save $300)
    4. **Prompt compression** (reduce context tokens by 30% = $67K savings)
    5. **Use local LLM for non-critical queries** (Llama-2: free)

    **Optimized cost: $110K/month (52% reduction)**

    ---

    ## Monitoring & Observability

    **Key Metrics to Track:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **End-to-End Latency (P95)** | < 3s | > 5s |
    | **Retrieval Latency (P95)** | < 100ms | > 500ms |
    | **LLM Latency (P95)** | < 2.5s | > 4s |
    | **Retrieval Recall@10** | > 90% | < 80% |
    | **Retrieval Precision@10** | > 80% | < 70% |
    | **Citation Accuracy** | 100% | < 99% |
    | **LLM Token Usage** | 2.5K/query | > 4K/query |
    | **Cost Per Query** | < $0.10 | > $0.20 |
    | **Cache Hit Rate** | > 20% | < 10% |

    ---

    ## LLM Observability

    ```python
    from typing import Dict
    import time
    import logging

    class LLMObservabilityWrapper:
        """
        Wrap LLM calls with observability
        Track latency, tokens, costs, errors
        """

        def __init__(self, llm_client, metrics_client):
            self.llm = llm_client
            self.metrics = metrics_client

        def generate(self, prompt: str, model: str = "gpt-4",
                    metadata: Dict = None) -> Dict:
            """
            Generate response with full observability

            Returns:
                {
                    "response": str,
                    "metrics": {
                        "latency_ms": float,
                        "tokens_used": int,
                        "cost_usd": float,
                        "model": str
                    }
                }
            """
            start_time = time.time()

            try:
                # Call LLM
                response = self.llm.complete(prompt, model=model)

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                tokens_used = response.usage.total_tokens
                cost_usd = self._calculate_cost(model, tokens_used)

                # Log metrics
                self.metrics.histogram("llm.latency_ms", latency_ms, tags={
                    "model": model,
                    "status": "success"
                })
                self.metrics.increment("llm.requests", tags={
                    "model": model,
                    "status": "success"
                })
                self.metrics.histogram("llm.tokens", tokens_used, tags={
                    "model": model
                })
                self.metrics.histogram("llm.cost_usd", cost_usd, tags={
                    "model": model
                })

                # Log to structured logger
                logging.info("LLM request", extra={
                    "model": model,
                    "latency_ms": latency_ms,
                    "tokens": tokens_used,
                    "cost_usd": cost_usd,
                    "metadata": metadata
                })

                return {
                    "response": response.text,
                    "metrics": {
                        "latency_ms": latency_ms,
                        "tokens_used": tokens_used,
                        "cost_usd": cost_usd,
                        "model": model
                    }
                }

            except Exception as e:
                # Log error
                self.metrics.increment("llm.requests", tags={
                    "model": model,
                    "status": "error",
                    "error_type": type(e).__name__
                })
                logging.error(f"LLM request failed: {e}", extra={
                    "model": model,
                    "metadata": metadata
                })
                raise

        def _calculate_cost(self, model: str, tokens: int) -> float:
            """Calculate cost based on model pricing"""
            pricing = {
                "gpt-4": 0.03 / 1000,
                "gpt-3.5-turbo": 0.002 / 1000,
                "claude-3-opus": 0.015 / 1000
            }
            return tokens * pricing.get(model, 0)
    ```

    ---

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Hybrid retrieval:** Semantic (vector) + keyword (BM25) for best recall
    2. **Multi-stage ranking:** Fast candidate generation (1000s) → reranking (100s) → final selection (10)
    3. **Chunking strategy:** Recursive character splitting with 10-20% overlap
    4. **Embedding model:** OpenAI ada-002 or Sentence-Transformers (cost vs quality trade-off)
    5. **Vector database:** Pinecone/Weaviate for ANN search (< 50ms latency)
    6. **Citation tracking:** Store chunk metadata, inject citation IDs into prompts
    7. **Conversation memory:** DynamoDB for multi-turn context (low latency, high availability)
    8. **Incremental indexing:** Kafka CDC for real-time document updates

    ---

    ## Interview Tips

    ✅ **Start with clarifying questions** - Document types? Query volume? Latency requirements?

    ✅ **Discuss chunking early** - Critical foundation that affects everything downstream

    ✅ **Explain retrieval trade-offs** - Semantic vs keyword vs hybrid, speed vs accuracy

    ✅ **Address hallucination prevention** - Prompt engineering, citation requirements, validation

    ✅ **Talk about costs** - LLM costs dominate (90%+), caching is critical

    ✅ **Mention real systems** - ChatGPT plugins, GitHub Copilot, Notion AI

    ✅ **Scalability matters** - Vector DB sharding, LLM batching, multi-level caching

    ✅ **Monitoring is essential** - Track retrieval quality, LLM costs, latency

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to prevent hallucinations?"** | Strict prompts ("only use provided context"), citation requirements, answer validation, confidence scores |
    | **"How to handle multi-hop reasoning?"** | Query decomposition, sub-query retrieval, iterative refinement, knowledge graph integration |
    | **"How to improve retrieval accuracy?"** | Hybrid search, reranking with cross-encoders, query expansion, fine-tuned embeddings |
    | **"How to reduce LLM costs?"** | Response caching (20% hit rate = huge savings), smaller models for simple queries, prompt compression, batching |
    | **"How to handle long documents?"** | Hierarchical chunking (summaries + details), map-reduce over chunks, section-aware retrieval |
    | **"How to maintain conversation context?"** | Conversation buffer (last N messages), conversation summary, context window management |
    | **"How to handle real-time updates?"** | Incremental indexing via Kafka CDC, upsert operations, cache invalidation |
    | **"How to scale to billions of documents?"** | Vector DB sharding, distributed embedding generation, multi-level caching, CDN for responses |
    | **"How to support multiple languages?"** | Multilingual embeddings (Cohere, mBERT), language detection, per-language indexes |
    | **"How to handle structured data?"** | Hybrid storage (text in vector DB, structured in SQL), metadata filtering, table understanding models |

    ---

    ## Real-World Examples

    **ChatGPT Retrieval Plugin:**
    - Uses OpenAI embeddings + Pinecone
    - Stores document chunks with metadata
    - Retrieves top-K chunks per query
    - Injects into GPT-4 context
    - Returns cited responses

    **GitHub Copilot:**
    - Indexes code repositories
    - Uses code-specific embeddings (CodeBERT)
    - Retrieves relevant code snippets
    - Generates code with LLM
    - Cites source files

    **Notion AI:**
    - Indexes user's workspace documents
    - Semantic search across notes, docs, wikis
    - Personalized responses based on user's content
    - Citation links to original pages
    - Real-time updates as documents change

    ---

    ## Advanced Topics to Mention

    - **Fine-tuned embeddings:** Train custom embeddings on domain-specific data
    - **Knowledge graphs:** Combine vector search with graph traversal for complex reasoning
    - **Adaptive retrieval:** Dynamically adjust top-K based on query complexity
    - **Query routing:** Route simple queries to cheaper models, complex to GPT-4
    - **Feedback loops:** Use user feedback to improve retrieval and ranking
    - **Guardrails:** Content filtering, PII detection, bias mitigation

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** OpenAI, Anthropic, LangChain, LlamaIndex, Enterprise AI, Cohere, Pinecone

---

*Master this problem and you'll be ready for: Enterprise AI chatbots, document Q&A systems, knowledge management platforms, AI-powered customer support, code assistants, research tools*
