# Design a Document Q&A System

A scalable document question-answering system that enables users to upload documents (PDFs, DOCX, etc.), ask natural language questions, and receive accurate answers with precise citations extracted from the document content.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100K documents, 10K queries/day, 500K document pages, 5K concurrent users |
| **Key Challenges** | PDF parsing (tables/images), intelligent chunking, semantic search accuracy, citation extraction, multi-document context |
| **Core Concepts** | Document parsing, text chunking, embeddings, vector search, retrieval augmented generation (RAG), citation mapping |
| **Companies** | ChatPDF, Claude.ai (Artifacts), Notion AI, Microsoft Copilot, LangChain, LlamaIndex, Enterprise AI platforms |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Document Upload** | Upload PDF, DOCX, TXT files (up to 50MB) | P0 (Must have) |
    | **Document Parsing** | Extract text, tables, images with layout preservation | P0 (Must have) |
    | **Question Answering** | Ask natural language questions about documents | P0 (Must have) |
    | **Citation Extraction** | Provide page numbers and exact text snippets for answers | P0 (Must have) |
    | **Multi-document QA** | Answer questions across multiple uploaded documents | P0 (Must have) |
    | **Semantic Search** | Find relevant sections using vector similarity | P0 (Must have) |
    | **Follow-up Questions** | Support conversational context with document | P1 (Should have) |
    | **Summarization** | Generate document summaries on demand | P1 (Should have) |
    | **Metadata Filtering** | Filter by document type, date, author, etc. | P1 (Should have) |
    | **Export Answers** | Export Q&A pairs with citations to PDF/DOCX | P2 (Nice to have) |
    | **OCR Support** | Extract text from scanned/image PDFs | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Document editing/annotation capabilities
    - Real-time collaborative document viewing
    - Advanced document versioning
    - Video/audio document processing
    - Custom model training/fine-tuning

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Upload Processing** | < 30s p95 for 20-page PDF | Users expect fast document ingestion |
    | **Query Latency** | < 3s p95 end-to-end | Acceptable for complex document search |
    | **Parsing Accuracy** | > 95% text extraction accuracy | Critical for answer quality |
    | **Citation Accuracy** | > 98% correct page/section mapping | Users must trust citations |
    | **Availability** | 99.9% uptime | Business-critical for enterprise users |
    | **Scalability** | Handle 10x document volume | Support growth without redesign |
    | **Cost Efficiency** | < $0.05 per document processed | Keep embedding/storage costs low |
    | **Security** | Document isolation, encryption at rest | Protect sensitive business documents |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 5,000
    Monthly Active Users (MAU): 15,000

    Document uploads:
    - Documents uploaded per DAU: 2 documents/day
    - Daily uploads: 5,000 × 2 = 10,000 documents/day
    - Average document size: 5 MB (≈ 20 pages)
    - Average pages per document: 20 pages
    - Daily pages processed: 10,000 × 20 = 200,000 pages/day

    Query traffic:
    - Queries per DAU: 2 queries/day
    - Daily queries: 5,000 × 2 = 10,000 queries/day
    - Query QPS: 10,000 / 86,400 ≈ 0.12 req/sec average
    - Peak QPS: 10x average = 1.2 req/sec (very manageable)

    Read/Write ratio: 1:1 (equal uploads and queries)
    ```

    ### Storage Estimates

    ```
    Document storage:
    - Total documents: 100,000 documents
    - Average document size: 5 MB
    - Raw storage: 100K × 5 MB = 500 GB

    Processed text/chunks:
    - Average text per document: 50 KB (extracted text)
    - Chunks per document: 40 chunks (1,000 tokens each)
    - Chunk storage: 100K × 50 KB = 5 GB

    Vector embeddings:
    - Chunks per document: 40 chunks
    - Embedding dimension: 1536 (OpenAI text-embedding-3-small)
    - Bytes per embedding: 1536 × 4 bytes (float32) = 6 KB
    - Total chunks: 100K docs × 40 chunks = 4M chunks
    - Embeddings storage: 4M × 6 KB = 24 GB

    Metadata:
    - Document metadata: 100K × 2 KB = 200 MB
    - Chunk metadata (page, position): 4M × 500 bytes = 2 GB

    Total: 500 GB (docs) + 5 GB (text) + 24 GB (embeddings) + 2.2 GB (metadata) ≈ 531 GB
    ```

    ### Compute Estimates

    ```
    Document processing (CPU):
    - PDF parsing: 10,000 docs/day × 2 seconds = 20,000 CPU-seconds/day
    - Text chunking: 10,000 docs/day × 1 second = 10,000 CPU-seconds/day
    - Daily CPU hours: 30,000 / 3,600 ≈ 8.3 CPU-hours/day
    - Concurrent workers: 5-10 workers (handle bursts)

    Embedding generation (GPU/API):
    - Chunks per day: 10,000 docs × 40 = 400,000 chunks/day
    - Embedding API calls: 400,000 calls/day (batch 100 per call = 4,000 API calls)
    - Cost: 4,000 calls × $0.0001 = $0.40/day ≈ $12/month (using OpenAI)

    Query processing:
    - Vector search: 10,000 queries/day × 50ms = 500 seconds/day (handled by vector DB)
    - LLM generation: 10,000 queries × 2 seconds = 20,000 seconds/day ≈ 5.5 hours
    - Token usage: 10,000 queries × (2,000 input + 500 output) = 25M tokens/day
    - Cost: 25M × $0.50/1M tokens = $12.50/day ≈ $375/month
    ```

    ### Memory Estimates

    ```
    Vector database cache:
    - Hot embeddings in memory: 1M chunks × 6 KB = 6 GB
    - Vector index overhead: +50% = 9 GB

    Document cache (recently accessed):
    - 1,000 recent documents × 5 MB = 5 GB

    LLM context cache:
    - Active queries: 10 concurrent × 50 KB = 500 KB

    Total memory: 9 GB (vector DB) + 5 GB (docs) + 0.5 GB (cache) ≈ 15 GB
    ```

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture

    ```mermaid
    graph TB
        subgraph "Client Layer"
            WebUI[Web UI]
            MobileApp[Mobile App]
            API_Client[API Client]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Rate Limiting, Auth]
        end

        subgraph "Application Services"
            UploadService[Upload Service<br/>Presigned URLs, Validation]
            QueryService[Query Service<br/>Question Processing]
            DocManager[Document Manager<br/>Metadata, Access Control]
        end

        subgraph "Document Processing Pipeline"
            Parser[Document Parser<br/>PyPDF2, PDFPlumber<br/>python-docx]
            Chunker[Chunking Engine<br/>Semantic Splitter<br/>RecursiveCharacterTextSplitter]
            Embedder[Embedding Service<br/>text-embedding-3-small<br/>Batch Processing]
        end

        subgraph "Data Storage"
            S3[Object Storage<br/>S3/MinIO<br/>Raw Documents]
            PostgreSQL[(PostgreSQL<br/>Document Metadata<br/>User Data)]
            VectorDB[(Vector Database<br/>Pinecone/Qdrant<br/>Weaviate)]
        end

        subgraph "Retrieval & Generation"
            Retriever[Retrieval Service<br/>Hybrid Search<br/>Reranking]
            LLM[LLM Service<br/>GPT-4/Claude<br/>RAG Pipeline]
            CitationExtractor[Citation Extractor<br/>Page Mapping<br/>Text Highlighting]
        end

        subgraph "Cache Layer"
            Redis[Redis<br/>Query Cache<br/>Embedding Cache]
        end

        subgraph "Async Processing"
            Queue[Message Queue<br/>SQS/RabbitMQ]
            Workers[Worker Pool<br/>Document Processing]
        end

        WebUI --> Gateway
        MobileApp --> Gateway
        API_Client --> Gateway

        Gateway --> UploadService
        Gateway --> QueryService
        Gateway --> DocManager

        UploadService --> S3
        UploadService --> Queue
        Queue --> Workers
        Workers --> Parser
        Parser --> Chunker
        Chunker --> Embedder
        Embedder --> VectorDB
        Embedder --> PostgreSQL

        QueryService --> Redis
        QueryService --> Retriever
        Retriever --> VectorDB
        Retriever --> LLM
        LLM --> CitationExtractor
        CitationExtractor --> S3
        CitationExtractor --> PostgreSQL

        DocManager --> PostgreSQL
        DocManager --> S3
    ```

    ---

    ## Component Responsibilities

    ### Document Processing Pipeline

    | Component | Responsibility | Technology |
    |-----------|----------------|------------|
    | **Document Parser** | Extract text, tables, images from PDFs/DOCX with layout preservation | PyPDF2, PDFPlumber, pdfminer.six, python-docx |
    | **Chunking Engine** | Split documents into semantic chunks (500-1500 tokens) with overlap | LangChain RecursiveCharacterTextSplitter, semantic chunking |
    | **Embedding Service** | Generate vector embeddings for each chunk, batch processing | OpenAI text-embedding-3-small, Cohere, Sentence-Transformers |

    ### Query Processing

    | Component | Responsibility | Technology |
    |-----------|----------------|------------|
    | **Retrieval Service** | Semantic search across chunks, hybrid search (vector + keyword), reranking | Pinecone, Qdrant, Weaviate, Cohere Rerank |
    | **LLM Service** | Generate answers using retrieved context, streaming responses | GPT-4, Claude, Llama 2, vLLM |
    | **Citation Extractor** | Map generated text back to source documents, extract page numbers and quotes | Custom text matching, fuzzy search, page index |

    ### Storage & Caching

    | Component | Responsibility | Technology |
    |-----------|----------------|------------|
    | **Object Storage** | Store raw document files, support large file uploads | AWS S3, Google Cloud Storage, MinIO |
    | **Vector Database** | Store and search embeddings with metadata filtering | Pinecone, Qdrant, Weaviate, Milvus |
    | **PostgreSQL** | Document metadata, user info, chunk mappings, citations | PostgreSQL with pg_vector extension |
    | **Redis** | Cache embeddings, query results, rate limiting | Redis with TTL-based eviction |

    ---

    ## API Design

    ### Document Upload API

    ```python
    POST /api/v1/documents/upload
    Content-Type: multipart/form-data

    Request:
    {
        "file": <binary>,
        "metadata": {
            "title": "Q4 2023 Financial Report",
            "author": "Finance Team",
            "document_type": "financial_report",
            "tags": ["finance", "quarterly", "2023"]
        }
    }

    Response (202 Accepted):
    {
        "document_id": "doc_abc123",
        "status": "processing",
        "estimated_time": "25s",
        "webhook_url": "/api/v1/documents/doc_abc123/status"
    }
    ```

    ### Get Document Status

    ```python
    GET /api/v1/documents/{document_id}/status

    Response:
    {
        "document_id": "doc_abc123",
        "status": "completed",  // processing, completed, failed
        "progress": 100,
        "metadata": {
            "pages": 45,
            "chunks": 90,
            "processing_time": "23s"
        },
        "error": null
    }
    ```

    ### Query API

    ```python
    POST /api/v1/query
    Content-Type: application/json

    Request:
    {
        "question": "What was the revenue growth in Q4 2023?",
        "document_ids": ["doc_abc123", "doc_def456"],  // optional, default: all user docs
        "filters": {
            "document_type": "financial_report",
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"}
        },
        "options": {
            "top_k": 5,                    // number of chunks to retrieve
            "include_citations": true,
            "stream": false
        }
    }

    Response:
    {
        "answer": "The revenue growth in Q4 2023 was 15.2% year-over-year...",
        "citations": [
            {
                "document_id": "doc_abc123",
                "document_title": "Q4 2023 Financial Report",
                "page_number": 3,
                "text_snippet": "Revenue increased by 15.2% YoY to $450M",
                "confidence": 0.95,
                "bounding_box": {"x": 100, "y": 200, "width": 400, "height": 50}
            }
        ],
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_xyz789",
                "text": "...",
                "score": 0.89,
                "page": 3
            }
        ],
        "metadata": {
            "query_time": "2.3s",
            "tokens_used": 2500
        }
    }
    ```

    ### Streaming Query API

    ```python
    POST /api/v1/query/stream
    Content-Type: application/json

    Request: (same as /query)

    Response (Server-Sent Events):
    event: chunk
    data: {"text": "The revenue growth", "citations": []}

    event: chunk
    data: {"text": " in Q4 2023 was", "citations": []}

    event: citation
    data: {"document_id": "doc_abc123", "page": 3, "text": "15.2% YoY"}

    event: chunk
    data: {"text": " 15.2% year-over-year", "citations": []}

    event: done
    data: {"metadata": {"query_time": "2.3s", "tokens_used": 2500}}
    ```

    ---

    ## Data Models

    ### Document Schema (PostgreSQL)

    ```sql
    CREATE TABLE documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id),
        title TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,  -- S3 path
        file_size_bytes BIGINT NOT NULL,
        mime_type TEXT NOT NULL,
        status TEXT NOT NULL,  -- processing, completed, failed
        pages INTEGER,
        total_chunks INTEGER,
        metadata JSONB,  -- custom metadata
        created_at TIMESTAMP DEFAULT NOW(),
        processed_at TIMESTAMP,
        INDEX idx_user_id (user_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    );

    CREATE TABLE document_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        page_number INTEGER,
        start_char INTEGER,  -- position in original document
        end_char INTEGER,
        embedding_id TEXT,  -- ID in vector database
        metadata JSONB,  -- chunk-specific metadata
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(document_id, chunk_index),
        INDEX idx_document_id (document_id),
        INDEX idx_page_number (page_number)
    );

    CREATE TABLE queries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id),
        question TEXT NOT NULL,
        answer TEXT,
        document_ids UUID[],
        retrieved_chunk_ids UUID[],
        citations JSONB,
        query_time_ms INTEGER,
        tokens_used INTEGER,
        created_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_user_id (user_id),
        INDEX idx_created_at (created_at)
    );
    ```

    ### Vector Database Schema (Qdrant)

    ```python
    {
        "collection_name": "document_chunks",
        "vectors": {
            "size": 1536,  # embedding dimension
            "distance": "Cosine"
        },
        "payload_schema": {
            "document_id": "keyword",
            "chunk_id": "keyword",
            "chunk_index": "integer",
            "page_number": "integer",
            "text": "text",
            "document_title": "text",
            "document_type": "keyword",
            "user_id": "keyword",
            "created_at": "datetime"
        }
    }
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Document Parsing & Text Extraction

    ### PDF Parsing Strategy

    Different PDF types require different parsing approaches:

    | PDF Type | Characteristics | Best Tool | Approach |
    |----------|----------------|-----------|----------|
    | **Text-based PDF** | Native text, searchable | PyPDF2, pdfplumber | Direct text extraction |
    | **Scanned PDF** | Images of text | Tesseract OCR, AWS Textract | OCR processing |
    | **Mixed PDF** | Text + tables + images | pdfplumber + OCR | Hybrid extraction |
    | **Complex Layout** | Multi-column, nested | PDFMiner.six, Unstructured | Layout-aware parsing |

    ### Implementation: Robust PDF Parser

    ```python
    from typing import List, Dict, Optional
    import pdfplumber
    from PyPDF2 import PdfReader
    from dataclasses import dataclass
    import pytesseract
    from PIL import Image
    import io

    @dataclass
    class ParsedPage:
        page_number: int
        text: str
        tables: List[List[List[str]]]
        images: List[Dict]
        metadata: Dict

    class DocumentParser:
        """
        Robust document parser supporting multiple PDF types.

        Strategy:
        1. Try text extraction (PyPDF2) - fast for text-based PDFs
        2. If text is sparse, use pdfplumber for better layout handling
        3. Extract tables separately using pdfplumber
        4. If still low quality, fall back to OCR
        """

        def __init__(self, ocr_enabled: bool = False):
            self.ocr_enabled = ocr_enabled
            self.min_text_threshold = 50  # minimum chars per page

        def parse_pdf(self, file_path: str) -> List[ParsedPage]:
            """Parse PDF and return structured pages with text, tables, images."""
            parsed_pages = []

            # Try fast extraction first
            reader = PdfReader(file_path)
            with pdfplumber.open(file_path) as pdf:
                for page_num, (pypdf_page, plumber_page) in enumerate(
                    zip(reader.pages, pdf.pages), start=1
                ):
                    # Extract text
                    text = self._extract_text(pypdf_page, plumber_page)

                    # Extract tables
                    tables = plumber_page.extract_tables()

                    # Convert tables to markdown for better LLM understanding
                    table_text = self._tables_to_markdown(tables)

                    # Extract images (for OCR if needed)
                    images = self._extract_images(plumber_page)

                    # OCR fallback if text is sparse
                    if len(text) < self.min_text_threshold and self.ocr_enabled:
                        ocr_text = self._ocr_page(plumber_page)
                        text = ocr_text if len(ocr_text) > len(text) else text

                    parsed_pages.append(ParsedPage(
                        page_number=page_num,
                        text=text + "\n\n" + table_text,
                        tables=tables,
                        images=images,
                        metadata={
                            "width": plumber_page.width,
                            "height": plumber_page.height,
                            "extraction_method": "text" if len(text) > self.min_text_threshold else "ocr"
                        }
                    ))

            return parsed_pages

        def _extract_text(self, pypdf_page, plumber_page) -> str:
            """Try multiple extraction methods, return best result."""
            # Method 1: PyPDF2 (fastest)
            text1 = pypdf_page.extract_text() or ""

            # Method 2: pdfplumber (better layout preservation)
            text2 = plumber_page.extract_text() or ""

            # Return longer text (usually higher quality)
            return text2 if len(text2) > len(text1) else text1

        def _tables_to_markdown(self, tables: List[List[List[str]]]) -> str:
            """Convert extracted tables to markdown format."""
            if not tables:
                return ""

            markdown_tables = []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Create markdown table
                md_table = []
                headers = table[0]
                md_table.append("| " + " | ".join(headers) + " |")
                md_table.append("| " + " | ".join(["---"] * len(headers)) + " |")

                for row in table[1:]:
                    md_table.append("| " + " | ".join(row) + " |")

                markdown_tables.append("\n".join(md_table))

            return "\n\n".join(markdown_tables)

        def _extract_images(self, page) -> List[Dict]:
            """Extract images from page with metadata."""
            images = []
            for img_obj in page.images:
                images.append({
                    "x": img_obj["x0"],
                    "y": img_obj["top"],
                    "width": img_obj["width"],
                    "height": img_obj["height"]
                })
            return images

        def _ocr_page(self, page) -> str:
            """OCR fallback for scanned pages."""
            # Convert page to image
            img = page.to_image(resolution=300)
            pil_img = img.original

            # Run OCR
            text = pytesseract.image_to_string(pil_img)
            return text

    # Usage
    parser = DocumentParser(ocr_enabled=True)
    parsed_pages = parser.parse_pdf("financial_report.pdf")

    for page in parsed_pages:
        print(f"Page {page.page_number}: {len(page.text)} chars, {len(page.tables)} tables")
    ```

    ---

    ## 3.2 Intelligent Chunking Strategies

    ### Why Chunking Matters

    | Challenge | Impact | Solution |
    |-----------|--------|----------|
    | **Context Window Limits** | LLMs have token limits (4K-128K) | Split documents into retrievable chunks |
    | **Retrieval Granularity** | Large chunks = poor precision | Smaller chunks (500-1500 tokens) |
    | **Semantic Coherence** | Random splits break meaning | Semantic-aware splitting |
    | **Citation Accuracy** | Need to map answers to sources | Track chunk position in original doc |

    ### Chunking Strategies Comparison

    ```python
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        SemanticChunker
    )
    from langchain_openai import OpenAIEmbeddings
    from typing import List, Dict

    class ChunkingEngine:
        """
        Multi-strategy document chunking with overlap for context preservation.

        Strategies:
        1. Fixed-size chunking: Simple, fast, good baseline
        2. Recursive chunking: Split by paragraphs/sentences, then by size
        3. Semantic chunking: Split at semantic boundaries (experimental)
        """

        def __init__(self, strategy: str = "recursive"):
            self.strategy = strategy
            self.chunk_size = 1000  # tokens
            self.chunk_overlap = 200  # tokens

        def chunk_document(
            self,
            parsed_pages: List[ParsedPage],
            document_id: str
        ) -> List[Dict]:
            """
            Chunk document with metadata tracking for citations.

            Returns chunks with:
            - text: chunk content
            - metadata: page_number, char_position, document_id
            """
            # Combine pages with page markers
            full_text = ""
            page_boundaries = []  # track where each page starts
            char_position = 0

            for page in parsed_pages:
                page_start = char_position
                page_text = f"\n\n--- Page {page.page_number} ---\n\n{page.text}"
                full_text += page_text
                char_position += len(page_text)

                page_boundaries.append({
                    "page_number": page.page_number,
                    "start_char": page_start,
                    "end_char": char_position
                })

            # Apply chunking strategy
            if self.strategy == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],  # priority order
                    length_function=self._token_length
                )
            elif self.strategy == "semantic":
                splitter = SemanticChunker(
                    embeddings=OpenAIEmbeddings(),
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=80
                )
            else:
                splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

            # Split text
            chunks = splitter.split_text(full_text)

            # Add metadata to each chunk
            chunk_objects = []
            current_position = 0

            for chunk_idx, chunk_text in enumerate(chunks):
                # Find chunk position in original text
                chunk_start = full_text.find(chunk_text, current_position)
                chunk_end = chunk_start + len(chunk_text)
                current_position = chunk_start + 1

                # Determine page number
                page_number = self._get_page_number(chunk_start, page_boundaries)

                chunk_objects.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "document_id": document_id,
                    "page_number": page_number,
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                    "metadata": {
                        "chunk_size_tokens": self._token_length(chunk_text),
                        "contains_table": "---" in chunk_text  # table marker
                    }
                })

            return chunk_objects

        def _token_length(self, text: str) -> int:
            """Estimate token count (rough approximation)."""
            # GPT-3/4 average: ~4 chars per token
            return len(text) // 4

        def _get_page_number(self, char_position: int, page_boundaries: List[Dict]) -> int:
            """Map character position to page number."""
            for page_info in page_boundaries:
                if page_info["start_char"] <= char_position < page_info["end_char"]:
                    return page_info["page_number"]
            return page_boundaries[-1]["page_number"]  # fallback

    # Usage
    chunker = ChunkingEngine(strategy="recursive")
    chunks = chunker.chunk_document(parsed_pages, document_id="doc_123")

    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['text'][:200]}...")
    ```

    ### Chunking Best Practices

    | Parameter | Recommended Value | Reasoning |
    |-----------|-------------------|-----------|
    | **Chunk Size** | 500-1500 tokens | Balance between context and precision |
    | **Overlap** | 10-20% of chunk size | Prevent splitting of key information |
    | **Separators** | `\n\n`, `\n`, `. ` | Split at natural boundaries |
    | **Max Chunks/Doc** | 100-500 chunks | Keep indexing time reasonable |

    ---

    ## 3.3 Embedding Generation & Vector Storage

    ### Embedding Strategy

    ```python
    from typing import List, Dict, Optional
    from openai import OpenAI
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import hashlib

    class EmbeddingService:
        """
        Generate and store embeddings with caching and batch processing.

        Features:
        - Batch processing (100 texts per API call)
        - Redis caching to avoid re-embedding
        - Vector database integration
        """

        def __init__(
            self,
            embedding_model: str = "text-embedding-3-small",
            vector_db_url: str = "localhost:6333"
        ):
            self.client = OpenAI()
            self.model = embedding_model
            self.dimension = 1536  # text-embedding-3-small dimension

            # Initialize vector database
            self.qdrant = QdrantClient(url=vector_db_url)
            self._ensure_collection()

        def _ensure_collection(self):
            """Create Qdrant collection if doesn't exist."""
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if "document_chunks" not in collection_names:
                self.qdrant.create_collection(
                    collection_name="document_chunks",
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )

        def embed_and_store_chunks(
            self,
            chunks: List[Dict],
            batch_size: int = 100
        ) -> List[str]:
            """
            Generate embeddings for chunks and store in vector database.

            Returns list of embedding IDs.
            """
            embedding_ids = []

            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]

                # Extract texts
                texts = [chunk["text"] for chunk in batch_chunks]

                # Generate embeddings (batched API call)
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )

                embeddings = [item.embedding for item in response.data]

                # Prepare points for Qdrant
                points = []
                for chunk, embedding in zip(batch_chunks, embeddings):
                    point_id = self._generate_id(chunk)

                    points.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_id": chunk["document_id"],
                            "chunk_index": chunk["chunk_index"],
                            "page_number": chunk["page_number"],
                            "text": chunk["text"],
                            "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"],
                            "metadata": chunk.get("metadata", {})
                        }
                    ))

                    embedding_ids.append(point_id)

                # Store in vector database
                self.qdrant.upsert(
                    collection_name="document_chunks",
                    points=points
                )

            return embedding_ids

        def _generate_id(self, chunk: Dict) -> str:
            """Generate deterministic ID for chunk."""
            key = f"{chunk['document_id']}_{chunk['chunk_index']}"
            return hashlib.sha256(key.encode()).hexdigest()

        def search_similar_chunks(
            self,
            query: str,
            document_ids: Optional[List[str]] = None,
            top_k: int = 5,
            score_threshold: float = 0.7
        ) -> List[Dict]:
            """
            Search for similar chunks using vector similarity.

            Returns chunks with similarity scores.
            """
            # Embed query
            response = self.client.embeddings.create(
                input=[query],
                model=self.model
            )
            query_embedding = response.data[0].embedding

            # Build filter
            query_filter = None
            if document_ids:
                query_filter = {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"any": document_ids}
                        }
                    ]
                }

            # Search vector database
            search_results = self.qdrant.search(
                collection_name="document_chunks",
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Format results
            chunks = []
            for result in search_results:
                chunks.append({
                    "chunk_id": result.id,
                    "score": result.score,
                    "text": result.payload["text"],
                    "document_id": result.payload["document_id"],
                    "page_number": result.payload["page_number"],
                    "start_char": result.payload["start_char"],
                    "end_char": result.payload["end_char"],
                    "metadata": result.payload.get("metadata", {})
                })

            return chunks

    # Usage
    embedding_service = EmbeddingService()

    # Embed and store chunks
    embedding_ids = embedding_service.embed_and_store_chunks(chunks)
    print(f"Stored {len(embedding_ids)} embeddings")

    # Search
    results = embedding_service.search_similar_chunks(
        query="What was the revenue growth?",
        document_ids=["doc_123"],
        top_k=5
    )

    for result in results:
        print(f"Score: {result['score']:.3f}, Page: {result['page_number']}")
        print(f"Text: {result['text'][:200]}...")
    ```

    ---

    ## 3.4 RAG Pipeline with Citation Extraction

    ### Complete RAG Implementation

    ```python
    from typing import List, Dict, Optional
    from openai import OpenAI
    import re
    from dataclasses import dataclass

    @dataclass
    class Citation:
        document_id: str
        document_title: str
        page_number: int
        text_snippet: str
        confidence: float
        start_char: int
        end_char: int

    class DocumentQASystem:
        """
        Complete Document Q&A system with RAG and citation extraction.

        Pipeline:
        1. Embed query
        2. Retrieve relevant chunks (vector search)
        3. Rerank chunks (optional)
        4. Generate answer with LLM
        5. Extract citations from answer
        """

        def __init__(
            self,
            embedding_service: EmbeddingService,
            model: str = "gpt-4-turbo-preview"
        ):
            self.embedding_service = embedding_service
            self.client = OpenAI()
            self.model = model

        def answer_question(
            self,
            question: str,
            document_ids: Optional[List[str]] = None,
            top_k: int = 5,
            include_citations: bool = True
        ) -> Dict:
            """
            Answer question using RAG pipeline with citations.
            """
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.embedding_service.search_similar_chunks(
                query=question,
                document_ids=document_ids,
                top_k=top_k
            )

            if not retrieved_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer this question.",
                    "citations": [],
                    "retrieved_chunks": []
                }

            # Step 2: Build context for LLM
            context = self._build_context(retrieved_chunks)

            # Step 3: Generate answer
            answer = self._generate_answer(question, context)

            # Step 4: Extract citations
            citations = []
            if include_citations:
                citations = self._extract_citations(answer, retrieved_chunks)

            return {
                "answer": answer,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks
            }

        def _build_context(self, chunks: List[Dict]) -> str:
            """Build context string with source markers for citation tracking."""
            context_parts = []

            for i, chunk in enumerate(chunks):
                # Add source marker for citation extraction
                source_marker = f"[SOURCE_{i}]"
                context_parts.append(
                    f"{source_marker}\n"
                    f"Document: {chunk['document_id']}\n"
                    f"Page: {chunk['page_number']}\n"
                    f"Content: {chunk['text']}\n"
                )

            return "\n".join(context_parts)

        def _generate_answer(self, question: str, context: str) -> str:
            """Generate answer using LLM with retrieved context."""
            system_prompt = """You are a helpful assistant that answers questions based on provided document excerpts.

Instructions:
1. Answer the question using ONLY the information from the provided sources
2. If the answer is not in the sources, say "I cannot find this information in the provided documents"
3. Be specific and cite information accurately
4. When referencing information, mention which source it came from (e.g., "According to SOURCE_0...")
5. If information appears in multiple sources, mention all relevant sources

Format your answer clearly and include source references inline."""

            user_prompt = f"""Question: {question}

Sources:
{context}

Answer the question using the sources above. Include source references in your answer."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # lower temperature for factual answers
                max_tokens=1000
            )

            return response.choices[0].message.content

        def _extract_citations(
            self,
            answer: str,
            retrieved_chunks: List[Dict]
        ) -> List[Citation]:
            """
            Extract citations by mapping source references in answer to chunks.
            """
            citations = []

            # Find all source references in answer (e.g., SOURCE_0, SOURCE_1)
            source_pattern = r'SOURCE_(\d+)'
            source_refs = re.findall(source_pattern, answer)

            for source_idx_str in source_refs:
                source_idx = int(source_idx_str)

                if source_idx < len(retrieved_chunks):
                    chunk = retrieved_chunks[source_idx]

                    # Extract relevant snippet (first 200 chars of chunk)
                    snippet = chunk["text"][:200]
                    if len(chunk["text"]) > 200:
                        snippet += "..."

                    citations.append(Citation(
                        document_id=chunk["document_id"],
                        document_title=chunk.get("document_title", "Unknown"),
                        page_number=chunk["page_number"],
                        text_snippet=snippet,
                        confidence=chunk["score"],
                        start_char=chunk["start_char"],
                        end_char=chunk["end_char"]
                    ))

            # Deduplicate citations (same page/document)
            unique_citations = []
            seen = set()

            for citation in citations:
                key = (citation.document_id, citation.page_number)
                if key not in seen:
                    seen.add(key)
                    unique_citations.append(citation)

            return unique_citations

        def answer_question_streaming(
            self,
            question: str,
            document_ids: Optional[List[str]] = None,
            top_k: int = 5
        ):
            """
            Stream answer generation for better UX.

            Yields chunks of text as they're generated.
            """
            # Retrieve chunks
            retrieved_chunks = self.embedding_service.search_similar_chunks(
                query=question,
                document_ids=document_ids,
                top_k=top_k
            )

            if not retrieved_chunks:
                yield {
                    "type": "answer",
                    "content": "I couldn't find relevant information to answer this question."
                }
                return

            # Build context
            context = self._build_context(retrieved_chunks)

            # System and user prompts (same as non-streaming)
            system_prompt = """You are a helpful assistant that answers questions based on provided document excerpts.

Instructions:
1. Answer the question using ONLY the information from the provided sources
2. If the answer is not in the sources, say "I cannot find this information in the provided documents"
3. Be specific and cite information accurately
4. When referencing information, mention which source it came from (e.g., "According to SOURCE_0...")"""

            user_prompt = f"""Question: {question}

Sources:
{context}

Answer the question using the sources above."""

            # Stream response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                stream=True
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content

                    yield {
                        "type": "chunk",
                        "content": content
                    }

            # Extract and yield citations after complete answer
            citations = self._extract_citations(full_answer, retrieved_chunks)

            yield {
                "type": "citations",
                "content": [
                    {
                        "document_id": c.document_id,
                        "page_number": c.page_number,
                        "text_snippet": c.text_snippet,
                        "confidence": c.confidence
                    }
                    for c in citations
                ]
            }

            yield {
                "type": "done",
                "content": {
                    "retrieved_chunks": len(retrieved_chunks)
                }
            }

    # Usage
    qa_system = DocumentQASystem(embedding_service)

    # Non-streaming
    result = qa_system.answer_question(
        question="What was the revenue growth in Q4?",
        document_ids=["doc_123"],
        top_k=5
    )

    print("Answer:", result["answer"])
    print(f"\nCitations ({len(result['citations'])}):")
    for citation in result["citations"]:
        print(f"  - Page {citation.page_number}: {citation.text_snippet}")

    # Streaming
    print("\n\nStreaming answer:")
    for event in qa_system.answer_question_streaming(
        question="What was the revenue growth in Q4?",
        document_ids=["doc_123"]
    ):
        if event["type"] == "chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "citations":
            print("\n\nCitations:")
            for citation in event["content"]:
                print(f"  - Page {citation['page_number']}")
    ```

    ---

    ## 3.5 Document Processing Workflow

    ### End-to-End Pipeline

    ```python
    from typing import Dict
    import asyncio
    from celery import Celery
    import boto3

    # Celery app for async processing
    celery_app = Celery('document_qa', broker='redis://localhost:6379/0')

    @celery_app.task(bind=True)
    def process_document_task(self, document_id: str, file_path: str):
        """
        Async task to process uploaded document.

        Steps:
        1. Parse document (extract text, tables, images)
        2. Chunk document into semantic pieces
        3. Generate embeddings for chunks
        4. Store in vector database
        5. Update document status
        """
        try:
            # Update status
            update_document_status(document_id, "processing", progress=10)

            # Step 1: Parse PDF
            parser = DocumentParser(ocr_enabled=True)
            parsed_pages = parser.parse_pdf(file_path)
            update_document_status(document_id, "processing", progress=30)

            # Step 2: Chunk document
            chunker = ChunkingEngine(strategy="recursive")
            chunks = chunker.chunk_document(parsed_pages, document_id)
            update_document_status(document_id, "processing", progress=50)

            # Step 3: Generate embeddings and store
            embedding_service = EmbeddingService()
            embedding_ids = embedding_service.embed_and_store_chunks(chunks)
            update_document_status(document_id, "processing", progress=80)

            # Step 4: Store chunk metadata in PostgreSQL
            store_chunks_metadata(document_id, chunks, embedding_ids)
            update_document_status(document_id, "processing", progress=90)

            # Step 5: Mark as completed
            update_document_status(
                document_id,
                "completed",
                progress=100,
                metadata={
                    "pages": len(parsed_pages),
                    "chunks": len(chunks),
                    "embeddings": len(embedding_ids)
                }
            )

            return {
                "status": "success",
                "document_id": document_id,
                "pages": len(parsed_pages),
                "chunks": len(chunks)
            }

        except Exception as e:
            # Mark as failed
            update_document_status(
                document_id,
                "failed",
                error=str(e)
            )
            raise

    def update_document_status(
        document_id: str,
        status: str,
        progress: int = None,
        metadata: Dict = None,
        error: str = None
    ):
        """Update document processing status in database."""
        # PostgreSQL update
        pass

    def store_chunks_metadata(
        document_id: str,
        chunks: List[Dict],
        embedding_ids: List[str]
    ):
        """Store chunk metadata in PostgreSQL for citation mapping."""
        # Bulk insert into document_chunks table
        pass

    # Upload handler
    @app.post("/api/v1/documents/upload")
    async def upload_document(file: UploadFile, metadata: Dict):
        """Handle document upload and trigger async processing."""
        # Generate document ID
        document_id = generate_id()

        # Upload to S3
        s3_path = f"documents/{document_id}/{file.filename}"
        s3 = boto3.client('s3')
        s3.upload_fileobj(file.file, "my-bucket", s3_path)

        # Create document record
        create_document_record(
            document_id=document_id,
            file_name=file.filename,
            file_path=s3_path,
            metadata=metadata,
            status="processing"
        )

        # Trigger async processing
        process_document_task.delay(document_id, s3_path)

        return {
            "document_id": document_id,
            "status": "processing",
            "estimated_time": "25s"
        }
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## 4.1 Performance Optimizations

    ### Caching Strategy

    | Cache Type | TTL | Storage | Use Case |
    |------------|-----|---------|----------|
    | **Embedding Cache** | 7 days | Redis | Cache embeddings for frequently accessed chunks |
    | **Query Result Cache** | 1 hour | Redis | Cache common questions and answers |
    | **Document Metadata Cache** | 24 hours | Redis | Cache document listings and metadata |
    | **Vector Index Cache** | Persistent | In-memory | Keep hot vectors in memory (Qdrant) |

    ### Embedding Cache Implementation

    ```python
    import redis
    import hashlib
    import json
    from typing import List, Optional

    class CachedEmbeddingService:
        """
        Embedding service with Redis caching to reduce API calls.

        Cache key: hash(text + model_name)
        Cache value: JSON-serialized embedding vector
        """

        def __init__(self, redis_url: str = "redis://localhost:6379"):
            self.redis = redis.from_url(redis_url)
            self.client = OpenAI()
            self.model = "text-embedding-3-small"
            self.cache_ttl = 7 * 24 * 3600  # 7 days

        def _cache_key(self, text: str) -> str:
            """Generate cache key from text."""
            content = f"{text}:{self.model}"
            return f"emb:{hashlib.sha256(content.encode()).hexdigest()}"

        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            """
            Embed texts with caching.

            1. Check cache for each text
            2. Only embed cache misses
            3. Store new embeddings in cache
            """
            embeddings = []
            texts_to_embed = []
            text_indices = []

            # Check cache
            for i, text in enumerate(texts):
                cache_key = self._cache_key(text)
                cached = self.redis.get(cache_key)

                if cached:
                    # Cache hit
                    embeddings.append(json.loads(cached))
                else:
                    # Cache miss - need to embed
                    embeddings.append(None)  # placeholder
                    texts_to_embed.append(text)
                    text_indices.append(i)

            # Embed cache misses (batched)
            if texts_to_embed:
                response = self.client.embeddings.create(
                    input=texts_to_embed,
                    model=self.model
                )

                # Store in cache and update results
                for i, (text, embedding_obj) in enumerate(
                    zip(texts_to_embed, response.data)
                ):
                    embedding = embedding_obj.embedding
                    original_index = text_indices[i]

                    # Update result
                    embeddings[original_index] = embedding

                    # Store in cache
                    cache_key = self._cache_key(text)
                    self.redis.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(embedding)
                    )

            return embeddings

        def get_cache_stats(self) -> Dict:
            """Get cache hit rate statistics."""
            # Track hits/misses using Redis counters
            hits = int(self.redis.get("emb:cache:hits") or 0)
            misses = int(self.redis.get("emb:cache:misses") or 0)
            total = hits + misses

            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total > 0 else 0,
                "total_requests": total
            }
    ```

    ### Query Result Caching

    ```python
    import hashlib
    import json
    from functools import wraps

    def cache_query_result(ttl: int = 3600):
        """
        Decorator to cache query results.

        Cache key: hash(question + document_ids + top_k)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, question: str, document_ids=None, top_k=5, **kwargs):
                # Generate cache key
                cache_key_content = f"{question}:{document_ids}:{top_k}"
                cache_key = f"query:{hashlib.sha256(cache_key_content.encode()).hexdigest()}"

                # Check cache
                redis_client = self.redis  # assumes self has redis client
                cached_result = redis_client.get(cache_key)

                if cached_result:
                    # Cache hit
                    redis_client.incr("query:cache:hits")
                    return json.loads(cached_result)

                # Cache miss - execute query
                redis_client.incr("query:cache:misses")
                result = func(self, question, document_ids, top_k, **kwargs)

                # Store in cache
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)  # default=str for datetime serialization
                )

                return result

            return wrapper
        return decorator

    # Usage in DocumentQASystem
    class OptimizedDocumentQASystem(DocumentQASystem):
        def __init__(self, *args, redis_url: str = "redis://localhost:6379", **kwargs):
            super().__init__(*args, **kwargs)
            self.redis = redis.from_url(redis_url)

        @cache_query_result(ttl=3600)
        def answer_question(self, question: str, document_ids=None, top_k=5, **kwargs):
            return super().answer_question(question, document_ids, top_k, **kwargs)
    ```

    ---

    ## 4.2 Batch Processing & Async Operations

    ### Batch Document Processing

    ```python
    from concurrent.futures import ThreadPoolExecutor
    from typing import List
    import time

    class BatchDocumentProcessor:
        """
        Process multiple documents in parallel with rate limiting.

        Features:
        - Parallel parsing (CPU-bound)
        - Batched embedding API calls
        - Progress tracking
        """

        def __init__(self, max_workers: int = 5):
            self.max_workers = max_workers
            self.parser = DocumentParser(ocr_enabled=True)
            self.chunker = ChunkingEngine()
            self.embedding_service = CachedEmbeddingService()

        def process_documents_batch(
            self,
            document_paths: List[str],
            document_ids: List[str]
        ) -> Dict:
            """Process multiple documents in parallel."""
            start_time = time.time()

            # Step 1: Parse documents in parallel (CPU-bound)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                parsed_results = list(executor.map(
                    self.parser.parse_pdf,
                    document_paths
                ))

            # Step 2: Chunk all documents
            all_chunks = []
            for parsed_pages, doc_id in zip(parsed_results, document_ids):
                chunks = self.chunker.chunk_document(parsed_pages, doc_id)
                all_chunks.extend(chunks)

            # Step 3: Batch embed all chunks (API-bound)
            # Group into optimal batch sizes (100 texts per API call)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                texts = [chunk["text"] for chunk in batch_chunks]

                # Single API call for batch
                embeddings = self.embedding_service.embed_texts(texts)
                all_embeddings.extend(embeddings)

            # Step 4: Store in vector database (batched upsert)
            self.embedding_service.store_embeddings_batch(all_chunks, all_embeddings)

            processing_time = time.time() - start_time

            return {
                "documents_processed": len(document_paths),
                "total_chunks": len(all_chunks),
                "processing_time": processing_time,
                "throughput": len(document_paths) / processing_time
            }
    ```

    ---

    ## 4.3 Streaming Response Optimization

    ### Server-Sent Events (SSE) Implementation

    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import json
    import asyncio

    app = FastAPI()

    @app.post("/api/v1/query/stream")
    async def stream_query(request: Request):
        """
        Stream query results using SSE.

        Event types:
        - chunk: text chunk from LLM
        - citation: citation found
        - done: processing complete
        """
        body = await request.json()
        question = body["question"]
        document_ids = body.get("document_ids")

        async def event_generator():
            qa_system = OptimizedDocumentQASystem(embedding_service)

            try:
                # Stream response
                for event in qa_system.answer_question_streaming(
                    question=question,
                    document_ids=document_ids
                ):
                    # Format as SSE
                    event_type = event["type"]
                    event_data = json.dumps(event["content"])

                    yield f"event: {event_type}\n"
                    yield f"data: {event_data}\n\n"

                    # Small delay to prevent overwhelming client
                    await asyncio.sleep(0.01)

            except Exception as e:
                # Send error event
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    # Client-side JavaScript example
    """
    const eventSource = new EventSource('/api/v1/query/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            question: "What was the revenue?",
            document_ids: ["doc_123"]
        })
    });

    eventSource.addEventListener('chunk', (e) => {
        const data = JSON.parse(e.data);
        answerDiv.textContent += data;  // append text chunk
    });

    eventSource.addEventListener('citation', (e) => {
        const citation = JSON.parse(e.data);
        citationsDiv.appendChild(createCitationElement(citation));
    });

    eventSource.addEventListener('done', (e) => {
        eventSource.close();
    });
    """
    ```

    ---

    ## 4.4 Metadata Filtering & Hybrid Search

    ### Advanced Retrieval with Filters

    ```python
    from typing import List, Dict, Optional
    from datetime import datetime

    class AdvancedRetriever:
        """
        Advanced retrieval with metadata filtering and hybrid search.

        Features:
        - Metadata filtering (date, author, document type)
        - Hybrid search (vector + keyword)
        - Reranking with cross-encoder
        """

        def __init__(self, qdrant_client, embedding_service):
            self.qdrant = qdrant_client
            self.embedding_service = embedding_service

        def search_with_filters(
            self,
            query: str,
            filters: Optional[Dict] = None,
            top_k: int = 10,
            use_hybrid: bool = True
        ) -> List[Dict]:
            """
            Search with metadata filters.

            Filters example:
            {
                "document_type": "financial_report",
                "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
                "author": "Finance Team",
                "tags": ["quarterly", "finance"]
            }
            """
            # Embed query
            query_embedding = self.embedding_service.embed_texts([query])[0]

            # Build Qdrant filter
            qdrant_filter = self._build_qdrant_filter(filters)

            # Vector search
            vector_results = self.qdrant.search(
                collection_name="document_chunks",
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k
            )

            # Hybrid search: combine with keyword search
            if use_hybrid:
                keyword_results = self._keyword_search(query, filters, top_k)
                results = self._merge_results(vector_results, keyword_results, top_k)
            else:
                results = vector_results

            # Rerank results
            reranked_results = self._rerank_results(query, results)

            return self._format_results(reranked_results)

        def _build_qdrant_filter(self, filters: Optional[Dict]) -> Optional[Dict]:
            """Build Qdrant filter from user filters."""
            if not filters:
                return None

            conditions = []

            # Document type filter
            if "document_type" in filters:
                conditions.append({
                    "key": "metadata.document_type",
                    "match": {"value": filters["document_type"]}
                })

            # Date range filter
            if "date_range" in filters:
                date_range = filters["date_range"]
                conditions.append({
                    "key": "metadata.created_at",
                    "range": {
                        "gte": date_range["start"],
                        "lte": date_range["end"]
                    }
                })

            # Tags filter (any of the tags)
            if "tags" in filters:
                conditions.append({
                    "key": "metadata.tags",
                    "match": {"any": filters["tags"]}
                })

            return {"must": conditions} if conditions else None

        def _keyword_search(
            self,
            query: str,
            filters: Optional[Dict],
            top_k: int
        ) -> List:
            """
            Keyword-based search using BM25.

            Can use Qdrant's full-text search or external Elasticsearch.
            """
            # Simplified: use Qdrant's payload text search
            # In production, consider Elasticsearch for better keyword search
            return self.qdrant.scroll(
                collection_name="document_chunks",
                scroll_filter={
                    "must": [
                        {
                            "key": "text",
                            "match": {"text": query}
                        }
                    ]
                },
                limit=top_k
            )[0]  # returns (points, next_offset)

        def _merge_results(
            self,
            vector_results: List,
            keyword_results: List,
            top_k: int
        ) -> List:
            """
            Merge vector and keyword search results.

            Strategy: Reciprocal Rank Fusion (RRF)
            Score = sum(1 / (k + rank)) for each result list
            """
            merged_scores = {}
            k = 60  # RRF constant

            # Add vector results
            for rank, result in enumerate(vector_results, start=1):
                doc_id = result.id
                merged_scores[doc_id] = merged_scores.get(doc_id, 0) + (1 / (k + rank))

            # Add keyword results
            for rank, result in enumerate(keyword_results, start=1):
                doc_id = result.id
                merged_scores[doc_id] = merged_scores.get(doc_id, 0) + (1 / (k + rank))

            # Sort by merged score
            sorted_ids = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

            # Get top_k results
            top_ids = [doc_id for doc_id, _ in sorted_ids[:top_k]]

            # Fetch full results
            all_results = {r.id: r for r in vector_results + keyword_results}
            return [all_results[doc_id] for doc_id in top_ids if doc_id in all_results]

        def _rerank_results(self, query: str, results: List) -> List:
            """
            Rerank results using cross-encoder model.

            Cross-encoder scores query-document pairs directly.
            More accurate but slower than bi-encoder (embeddings).
            """
            # Use Cohere Rerank API or local cross-encoder model
            from cohere import Client

            cohere = Client(api_key="your-api-key")

            # Prepare documents for reranking
            documents = [result.payload["text"] for result in results]

            # Rerank
            rerank_response = cohere.rerank(
                query=query,
                documents=documents,
                top_n=len(documents),
                model="rerank-english-v2.0"
            )

            # Reorder results based on reranking scores
            reranked_results = []
            for result in rerank_response.results:
                original_result = results[result.index]
                original_result.score = result.relevance_score  # update score
                reranked_results.append(original_result)

            return reranked_results

        def _format_results(self, results: List) -> List[Dict]:
            """Format results for API response."""
            return [
                {
                    "chunk_id": r.id,
                    "score": r.score,
                    "text": r.payload["text"],
                    "document_id": r.payload["document_id"],
                    "page_number": r.payload["page_number"],
                    "metadata": r.payload.get("metadata", {})
                }
                for r in results
            ]

    # Usage
    retriever = AdvancedRetriever(qdrant_client, embedding_service)

    results = retriever.search_with_filters(
        query="What was the revenue growth?",
        filters={
            "document_type": "financial_report",
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "tags": ["quarterly"]
        },
        top_k=10,
        use_hybrid=True
    )
    ```

    ---

    ## 4.5 Monitoring & Observability

    ### Key Metrics to Track

    | Metric | Target | Alert Threshold | Purpose |
    |--------|--------|-----------------|---------|
    | **Document Processing Time** | < 30s p95 | > 60s | Monitor parsing performance |
    | **Query Latency** | < 3s p95 | > 5s | User experience |
    | **Embedding Cache Hit Rate** | > 70% | < 50% | Cost optimization |
    | **Vector Search Latency** | < 100ms p95 | > 200ms | Retrieval performance |
    | **LLM Generation Time** | < 2s p95 | > 4s | Answer generation speed |
    | **Citation Accuracy** | > 95% | < 90% | Quality metric |
    | **Error Rate** | < 0.1% | > 1% | System reliability |

    ### Logging & Tracing

    ```python
    import structlog
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from prometheus_client import Counter, Histogram, Gauge

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )

    logger = structlog.get_logger()

    # Prometheus metrics
    query_counter = Counter('document_qa_queries_total', 'Total queries processed')
    query_latency = Histogram('document_qa_query_latency_seconds', 'Query latency')
    cache_hits = Counter('document_qa_cache_hits_total', 'Cache hits')
    cache_misses = Counter('document_qa_cache_misses_total', 'Cache misses')
    active_queries = Gauge('document_qa_active_queries', 'Active queries')

    # Instrumented query function
    @query_latency.time()
    def instrumented_answer_question(question: str, document_ids: List[str]):
        query_counter.inc()
        active_queries.inc()

        try:
            with tracer.start_as_current_span("answer_question") as span:
                span.set_attribute("question_length", len(question))
                span.set_attribute("num_documents", len(document_ids or []))

                logger.info(
                    "query.started",
                    question_length=len(question),
                    document_ids=document_ids
                )

                # Execute query
                result = qa_system.answer_question(
                    question=question,
                    document_ids=document_ids
                )

                logger.info(
                    "query.completed",
                    answer_length=len(result["answer"]),
                    num_citations=len(result["citations"]),
                    num_retrieved_chunks=len(result["retrieved_chunks"])
                )

                return result
        finally:
            active_queries.dec()
    ```

    ---

    ## 4.6 Cost Optimization

    ### Cost Breakdown (Monthly, 10K queries/day)

    | Component | Cost Driver | Monthly Cost | Optimization Strategy |
    |-----------|-------------|--------------|----------------------|
    | **Embedding API** | 400K chunks/day × $0.0001 | $12 | Cache embeddings, batch processing |
    | **LLM Generation** | 25M tokens/day × $0.50/1M | $375 | Cache common queries, use smaller models |
    | **Vector Database** | Qdrant Cloud (4M vectors) | $100 | Self-host on EC2, use compression |
    | **Object Storage** | 500GB S3 | $12 | Lifecycle policies, compress PDFs |
    | **Compute** | 5 workers × $0.10/hour | $360 | Spot instances, auto-scaling |
    | **Redis Cache** | 16GB ElastiCache | $50 | Use smaller instance, TTL tuning |
    | **Total** | | **$909/month** | |

    ### Cost Optimization Strategies

    ```python
    # 1. Use smaller embedding models
    # text-embedding-3-small: $0.02/1M tokens
    # text-embedding-3-large: $0.13/1M tokens
    # Savings: 85% on embeddings

    # 2. Aggressive caching
    cache_hit_rate = 0.7  # 70% cache hit rate
    monthly_embedding_cost = 12 * (1 - cache_hit_rate)
    # Savings: $8.40/month on embeddings

    # 3. Use cheaper LLM for simple queries
    def choose_model(query_complexity: str) -> str:
        """Route to appropriate model based on complexity."""
        if query_complexity == "simple":
            return "gpt-3.5-turbo"  # $0.50/1M tokens
        else:
            return "gpt-4-turbo"  # $10/1M tokens

    # 80% simple queries -> 80% savings on LLM costs
    monthly_llm_cost = 375 * 0.2 + 375 * 0.8 * 0.05
    # New cost: $90/month (vs $375)

    # 4. Self-host vector database
    # Qdrant on EC2 t3.xlarge (4 vCPU, 16GB RAM): $120/month
    # vs Qdrant Cloud: $100/month
    # Consider self-hosting at larger scale

    # Total optimized cost: ~$450/month (50% reduction)
    ```

=== "🎯 Key Takeaways"

    ## System Design Principles

    1. **Document Processing Pipeline**
        - Use robust parsing (PyPDF2 + pdfplumber + OCR fallback)
        - Semantic chunking with overlap preserves context
        - Batch processing reduces API costs and latency

    2. **Retrieval Strategy**
        - Vector search for semantic similarity
        - Hybrid search (vector + keyword) improves accuracy
        - Reranking with cross-encoder for final results
        - Metadata filtering for precise targeting

    3. **Citation Extraction**
        - Track chunk position in original document
        - Map generated text back to source pages
        - Provide precise page numbers and text snippets
        - Build user trust with accurate citations

    4. **Scalability Patterns**
        - Async document processing with job queues
        - Caching at multiple layers (embeddings, queries, results)
        - Streaming responses for better UX
        - Batch API calls to reduce costs

    5. **Quality Optimizations**
        - Layout-aware PDF parsing (tables, multi-column)
        - Semantic chunking over fixed-size splitting
        - Context preservation with chunk overlap
        - Reranking for higher precision

    ---

    ## Common Pitfalls to Avoid

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Naive chunking** | Poor retrieval accuracy | Use RecursiveCharacterTextSplitter with semantic boundaries |
    | **No chunk overlap** | Context loss at boundaries | 10-20% overlap between chunks |
    | **Ignoring document structure** | Tables/lists corrupted | Layout-aware parsing with pdfplumber |
    | **Large chunks** | Poor precision | Keep chunks 500-1500 tokens |
    | **No citation tracking** | Can't verify answers | Track chunk position, page numbers |
    | **Synchronous processing** | Slow uploads | Async processing with job queues |
    | **No caching** | High API costs | Cache embeddings, queries, results |
    | **Vector-only search** | Misses exact keyword matches | Hybrid search (vector + BM25) |

    ---

    ## Interview Tips

    ### What to Discuss

    1. **Requirements Phase**
        - Document types supported (PDF, DOCX, scanned)
        - Query types (factual, summarization, comparison)
        - Citation requirements (page numbers, text highlighting)
        - Scale parameters (documents, queries, users)

    2. **Design Phase**
        - Document processing pipeline (parse → chunk → embed)
        - RAG architecture (retrieve → rerank → generate)
        - Storage strategy (object storage + vector DB + metadata DB)
        - API design (sync vs async, streaming)

    3. **Deep Dive Phase**
        - PDF parsing challenges (tables, multi-column, scanned)
        - Chunking strategies (fixed vs semantic vs recursive)
        - Retrieval techniques (vector, hybrid, reranking)
        - Citation extraction algorithms

    4. **Scale Phase**
        - Caching strategy (embeddings, queries)
        - Batch processing for documents
        - Query optimization (metadata filtering)
        - Cost optimization (model selection, caching)

    ### Questions to Ask Interviewer

    - What document types are most common? (affects parsing strategy)
    - What's the average document length? (affects chunking)
    - Are citations required? How precise? (affects architecture)
    - Is multi-document QA required? (affects retrieval complexity)
    - What's the query latency SLA? (affects caching strategy)
    - OCR required for scanned documents? (affects cost/latency)

    ---

    ## Related Systems

    | System | Similarity | Key Differences |
    |--------|------------|-----------------|
    | **RAG System** | Core retrieval + generation pattern | Document Q&A focuses on citations and PDF parsing |
    | **Semantic Search** | Vector-based retrieval | Q&A adds LLM generation layer |
    | **ChatGPT** | LLM-powered chat | Document Q&A grounds answers in specific documents |
    | **Search Engine** | Information retrieval | Q&A generates natural language answers vs links |

    ---

    ## References & Resources

    ### Key Technologies

    - **Document Parsing:** [PyPDF2](https://pypdf2.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber), [Unstructured](https://unstructured.io/)
    - **Embeddings:** [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings), [Sentence-Transformers](https://www.sbert.net/)
    - **Vector Databases:** [Qdrant](https://qdrant.tech/), [Pinecone](https://www.pinecone.io/), [Weaviate](https://weaviate.io/)
    - **LLM Frameworks:** [LangChain](https://python.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/)

    ### Production Examples

    - **ChatPDF:** PDF-specific Q&A system
    - **Claude.ai Artifacts:** Document analysis and Q&A
    - **Notion AI:** Workspace document Q&A
    - **Microsoft Copilot:** Enterprise document intelligence

    ### Papers & Blogs

    - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
    - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
    - [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
    - [LangChain RAG Best Practices](https://blog.langchain.dev/rag/)

