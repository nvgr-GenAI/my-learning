# RAG Implementation: From Code to Production

!!! tip "🚀 Building Your RAG System"
    Ready to build your own RAG system? This comprehensive guide takes you from your first lines of code to production-ready implementations. Let's build something amazing!

## 🎯 The Implementation Journey

### 📖 Our Building Story

Think of building a RAG system like constructing a smart library:

=== "🏗️ The Construction Plan"

    **Phase 1: Foundation (Basic RAG)**
    - 📚 Set up document processing
    - 🔤 Create embeddings
    - 🗄️ Build vector storage
    - 🔍 Implement retrieval
    - 🤖 Connect to LLM

    **Phase 2: Enhancement (Advanced RAG)**
    - 🎯 Add query rewriting
    - 🔄 Implement reranking
    - 📊 Add evaluation metrics
    - ⚡ Optimize performance

    **Phase 3: Production (Enterprise RAG)**
    - 🛡️ Add monitoring
    - 🔒 Implement security
    - 📈 Scale for load
    - 🔄 Add CI/CD

## 🛠️ Implementation Patterns

### 🌟 Pattern 1: Basic RAG Implementation

Let's start with a complete, working RAG system:

=== "🐍 Python Implementation"

    ```python
    import os
    from typing import List, Dict, Any
    from dataclasses import dataclass
    from pathlib import Path
    
    # Core imports
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    
    @dataclass
    class RAGConfig:
        """Configuration for RAG system"""
        chunk_size: int = 1000
        chunk_overlap: int = 200
        embedding_model: str = "text-embedding-ada-002"
        llm_model: str = "gpt-3.5-turbo"
        retrieval_k: int = 5
        vector_store_path: str = "./chroma_db"
    
    class DocumentProcessor:
        """Handles document loading and chunking"""
        
        def __init__(self, config: RAGConfig):
            self.config = config
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
        def load_documents(self, file_paths: List[str]) -> List[Dict]:
            """Load documents from various file types"""
            documents = []
            
            for file_path in file_paths:
                path = Path(file_path)
                
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif path.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(file_path)
                else:
                    print(f"Unsupported file type: {path.suffix}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
            
            return documents
        
        def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
            """Split documents into chunks"""
            return self.text_splitter.split_documents(documents)
    
    class VectorStore:
        """Manages vector storage and retrieval"""
        
        def __init__(self, config: RAGConfig):
            self.config = config
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model
            )
            self.vector_store = None
        
        def create_vector_store(self, documents: List[Dict]) -> None:
            """Create and persist vector store"""
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.vector_store_path
            )
            self.vector_store.persist()
        
        def load_vector_store(self) -> None:
            """Load existing vector store"""
            if Path(self.config.vector_store_path).exists():
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store_path,
                    embedding_function=self.embeddings
                )
            else:
                raise FileNotFoundError("Vector store not found")
        
        def add_documents(self, documents: List[Dict]) -> None:
            """Add new documents to existing vector store"""
            if self.vector_store is None:
                self.load_vector_store()
            
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
    
    class RAGSystem:
        """Main RAG system orchestrator"""
        
        def __init__(self, config: RAGConfig):
            self.config = config
            self.processor = DocumentProcessor(config)
            self.vector_store = VectorStore(config)
            self.llm = OpenAI(model_name=config.llm_model)
            self.qa_chain = None
        
        def build_knowledge_base(self, file_paths: List[str]) -> None:
            """Build the knowledge base from documents"""
            print("🔄 Loading documents...")
            documents = self.processor.load_documents(file_paths)
            
            print("✂️ Chunking documents...")
            chunks = self.processor.chunk_documents(documents)
            
            print("🔤 Creating embeddings and vector store...")
            self.vector_store.create_vector_store(chunks)
            
            print("✅ Knowledge base built successfully!")
        
        def load_knowledge_base(self) -> None:
            """Load existing knowledge base"""
            try:
                self.vector_store.load_vector_store()
                print("✅ Knowledge base loaded successfully!")
            except FileNotFoundError:
                print("❌ Knowledge base not found. Please build it first.")
                raise
        
        def create_qa_chain(self) -> None:
            """Create the question-answering chain"""
            if self.vector_store.vector_store is None:
                self.load_knowledge_base()
            
            retriever = self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
        
        def ask_question(self, question: str) -> Dict[str, Any]:
            """Ask a question and get an answer with sources"""
            if self.qa_chain is None:
                self.create_qa_chain()
            
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown")
                    }
                    for doc in result["source_documents"]
                ]
            }
    
    # Usage Example
    def main():
        # Configuration
        config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3
        )
        
        # Initialize RAG system
        rag = RAGSystem(config)
        
        # Build knowledge base (run once)
        document_paths = [
            "path/to/your/documents/doc1.pdf",
            "path/to/your/documents/doc2.txt",
            "path/to/your/documents/doc3.md"
        ]
        
        try:
            rag.build_knowledge_base(document_paths)
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            return
        
        # Ask questions
        questions = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the recommendations mentioned?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            try:
                result = rag.ask_question(question)
                print(f"🤖 Answer: {result['answer']}")
                print(f"📚 Sources: {len(result['sources'])} documents")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    if __name__ == "__main__":
        main()
    ```

=== "🔧 Alternative: LlamaIndex Implementation"

    ```python
    from llama_index import (
        VectorStoreIndex, 
        SimpleDirectoryReader, 
        ServiceContext,
        StorageContext,
        load_index_from_storage
    )
    from llama_index.llms import OpenAI
    from llama_index.embeddings import OpenAIEmbedding
    
    class SimpleLlamaRAG:
        """Simplified RAG using LlamaIndex"""
        
        def __init__(self, data_dir: str, persist_dir: str = "./storage"):
            self.data_dir = data_dir
            self.persist_dir = persist_dir
            self.index = None
            self.query_engine = None
            
            # Configure service context
            llm = OpenAI(model="gpt-3.5-turbo")
            embed_model = OpenAIEmbedding()
            
            self.service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model
            )
        
        def build_index(self):
            """Build index from documents"""
            # Load documents
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents, 
                service_context=self.service_context
            )
            
            # Persist index
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        
        def load_index(self):
            """Load existing index"""
            storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir
            )
            self.index = load_index_from_storage(
                storage_context,
                service_context=self.service_context
            )
        
        def query(self, question: str) -> str:
            """Query the index"""
            if self.index is None:
                self.load_index()
            
            if self.query_engine is None:
                self.query_engine = self.index.as_query_engine()
            
            response = self.query_engine.query(question)
            return response.response
    
    # Usage
    rag = SimpleLlamaRAG("./documents")
    rag.build_index()
    answer = rag.query("What is the main topic?")
    print(answer)
    ```

### 🎯 Pattern 2: Advanced RAG with Query Enhancement

=== "🔄 Query Rewriting & Expansion"

    ```python
    import re
    from typing import List, Dict
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    
    class QueryEnhancer:
        """Enhances queries for better retrieval"""
        
        def __init__(self, llm):
            self.llm = llm
            self.query_rewrite_prompt = PromptTemplate(
                input_variables=["query"],
                template="""
                You are a query optimization expert. Given a user query, 
                rewrite it to be more effective for document retrieval.
                
                Original query: {query}
                
                Rewritten query (be specific and clear):
                """
            )
        
        def rewrite_query(self, query: str) -> str:
            """Rewrite query for better retrieval"""
            prompt = self.query_rewrite_prompt.format(query=query)
            return self.llm(prompt).strip()
        
        def expand_query(self, query: str) -> List[str]:
            """Generate multiple query variations"""
            expansion_prompt = f"""
            Generate 3 different variations of this query that might help 
            find relevant information:
            
            Original: {query}
            
            Variations:
            1.
            2.
            3.
            """
            
            response = self.llm(expansion_prompt)
            variations = re.findall(r'\d+\.\s*(.+)', response)
            return [var.strip() for var in variations]
    
    class HybridRetriever:
        """Combines multiple retrieval methods"""
        
        def __init__(self, vector_store, bm25_retriever, config):
            self.vector_store = vector_store
            self.bm25_retriever = bm25_retriever
            self.config = config
        
        def retrieve(self, query: str, k: int = 5) -> List[Dict]:
            """Hybrid retrieval combining vector and keyword search"""
            # Vector search
            vector_results = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            
            # Keyword search
            keyword_results = self.bm25_retriever.get_relevant_documents(query)
            
            # Combine and rank results
            combined_results = self._combine_results(
                vector_results, keyword_results, query
            )
            
            return combined_results[:k]
        
        def _combine_results(self, vector_results, keyword_results, query):
            """Combine and rank results from different retrievers"""
            # Implementation depends on your ranking strategy
            # This is a simplified example
            all_results = []
            
            # Add vector results with scores
            for doc, score in vector_results:
                all_results.append({
                    'document': doc,
                    'vector_score': score,
                    'keyword_score': 0,
                    'combined_score': score
                })
            
            # Add keyword results
            for doc in keyword_results:
                # Find if already in vector results
                found = False
                for result in all_results:
                    if result['document'].page_content == doc.page_content:
                        result['keyword_score'] = 1.0
                        result['combined_score'] = result['vector_score'] + 0.5
                        found = True
                        break
                
                if not found:
                    all_results.append({
                        'document': doc,
                        'vector_score': 0,
                        'keyword_score': 1.0,
                        'combined_score': 0.5
                    })
            
            # Sort by combined score
            all_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return [r['document'] for r in all_results]
    
    class RerankerModel:
        """Reranks retrieved documents for relevance"""
        
        def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        
        def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
            """Rerank documents based on relevance to query"""
            # Prepare query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k documents
            return [doc for doc, score in doc_scores[:top_k]]
    
    class AdvancedRAGSystem:
        """Advanced RAG with query enhancement and reranking"""
        
        def __init__(self, config: RAGConfig):
            self.config = config
            self.llm = OpenAI(model_name=config.llm_model)
            self.query_enhancer = QueryEnhancer(self.llm)
            self.reranker = RerankerModel()
            self.vector_store = None
            self.hybrid_retriever = None
        
        def ask_question(self, question: str) -> Dict[str, Any]:
            """Enhanced question answering with reranking"""
            # Step 1: Enhance query
            enhanced_query = self.query_enhancer.rewrite_query(question)
            query_variations = self.query_enhancer.expand_query(question)
            
            # Step 2: Retrieve documents
            all_documents = []
            
            # Original query
            docs = self.hybrid_retriever.retrieve(question, k=10)
            all_documents.extend(docs)
            
            # Enhanced query
            docs = self.hybrid_retriever.retrieve(enhanced_query, k=10)
            all_documents.extend(docs)
            
            # Query variations
            for variation in query_variations:
                docs = self.hybrid_retriever.retrieve(variation, k=5)
                all_documents.extend(docs)
            
            # Remove duplicates
            unique_docs = self._remove_duplicates(all_documents)
            
            # Step 3: Rerank documents
            reranked_docs = self.reranker.rerank(
                question, unique_docs, top_k=self.config.retrieval_k
            )
            
            # Step 4: Generate answer
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            answer_prompt = f"""
            Based on the following context, answer the question comprehensively:
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            answer = self.llm(answer_prompt)
            
            return {
                "answer": answer.strip(),
                "original_query": question,
                "enhanced_query": enhanced_query,
                "query_variations": query_variations,
                "sources": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown")
                    }
                    for doc in reranked_docs
                ]
            }
        
        def _remove_duplicates(self, documents):
            """Remove duplicate documents"""
            seen = set()
            unique_docs = []
            
            for doc in documents:
                content_hash = hash(doc.page_content)
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_docs.append(doc)
            
            return unique_docs
    ```

### 🏭 Pattern 3: Production-Ready RAG

=== "🛡️ Enterprise RAG Implementation"

    ```python
    import asyncio
    import logging
    from typing import List, Dict, Any, Optional
    from dataclasses import dataclass
    from contextlib import asynccontextmanager
    
    # Monitoring and observability
    from prometheus_client import Counter, Histogram, Gauge
    import structlog
    
    # Caching
    from redis import Redis
    import pickle
    
    # Rate limiting
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    # FastAPI for production API
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    
    # Metrics
    query_counter = Counter('rag_queries_total', 'Total RAG queries')
    query_duration = Histogram('rag_query_duration_seconds', 'Query duration')
    active_connections = Gauge('rag_active_connections', 'Active connections')
    
    # Structured logging
    logger = structlog.get_logger()
    
    @dataclass
    class ProductionRAGConfig:
        """Production configuration with all the bells and whistles"""
        # Core RAG settings
        chunk_size: int = 1000
        chunk_overlap: int = 200
        retrieval_k: int = 5
        
        # Performance settings
        max_concurrent_queries: int = 100
        query_timeout: int = 30
        cache_ttl: int = 3600  # 1 hour
        
        # Monitoring
        log_level: str = "INFO"
        metrics_enabled: bool = True
        
        # Security
        api_key_required: bool = True
        rate_limit: str = "100/minute"
        
        # Storage
        vector_store_type: str = "chroma"  # chroma, pinecone, weaviate
        redis_url: str = "redis://localhost:6379"
        
        # Model settings
        embedding_model: str = "text-embedding-ada-002"
        llm_model: str = "gpt-3.5-turbo"
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    class CacheManager:
        """Manages caching for RAG responses"""
        
        def __init__(self, redis_url: str, ttl: int = 3600):
            self.redis = Redis.from_url(redis_url)
            self.ttl = ttl
        
        def get_cached_response(self, query: str) -> Optional[Dict]:
            """Get cached response for query"""
            try:
                cached = self.redis.get(f"rag_query:{hash(query)}")
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning("Cache get failed", error=str(e))
            return None
        
        def cache_response(self, query: str, response: Dict) -> None:
            """Cache response for query"""
            try:
                self.redis.setex(
                    f"rag_query:{hash(query)}",
                    self.ttl,
                    pickle.dumps(response)
                )
            except Exception as e:
                logger.warning("Cache set failed", error=str(e))
    
    class SecurityManager:
        """Handles authentication and authorization"""
        
        def __init__(self, api_keys: List[str]):
            self.api_keys = set(api_keys)
        
        def verify_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
            """Verify API key"""
            return credentials.credentials in self.api_keys
    
    class ProductionRAGSystem:
        """Production-ready RAG system with monitoring, caching, and security"""
        
        def __init__(self, config: ProductionRAGConfig):
            self.config = config
            self.cache_manager = CacheManager(config.redis_url, config.cache_ttl)
            self.security_manager = SecurityManager(["your-api-key-here"])
            
            # Initialize core components
            self._initialize_components()
            
            # Setup monitoring
            if config.metrics_enabled:
                self._setup_monitoring()
        
        def _initialize_components(self):
            """Initialize RAG components"""
            # This would initialize your vector store, LLM, etc.
            # Implementation depends on your chosen components
            pass
        
        def _setup_monitoring(self):
            """Setup monitoring and logging"""
            logging.basicConfig(level=self.config.log_level)
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        async def ask_question(
            self, 
            question: str, 
            user_id: str = None,
            session_id: str = None
        ) -> Dict[str, Any]:
            """Ask question with full production features"""
            request_id = f"req_{hash(question + str(user_id))}"
            
            # Start monitoring
            query_counter.inc()
            active_connections.inc()
            
            logger.info(
                "Query started",
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                question_length=len(question)
            )
            
            try:
                with query_duration.time():
                    # Check cache first
                    cached_response = self.cache_manager.get_cached_response(question)
                    if cached_response:
                        logger.info("Cache hit", request_id=request_id)
                        return cached_response
                    
                    # Process query
                    response = await self._process_query(question, request_id)
                    
                    # Cache response
                    self.cache_manager.cache_response(question, response)
                    
                    logger.info(
                        "Query completed",
                        request_id=request_id,
                        sources_count=len(response.get("sources", [])),
                        answer_length=len(response.get("answer", ""))
                    )
                    
                    return response
                    
            except Exception as e:
                logger.error(
                    "Query failed",
                    request_id=request_id,
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                active_connections.dec()
        
        async def _process_query(self, question: str, request_id: str) -> Dict[str, Any]:
            """Process the actual query"""
            # This would contain your RAG logic
            # For now, returning a mock response
            return {
                "answer": "This is a mock answer for production testing",
                "sources": [],
                "request_id": request_id,
                "timestamp": "2024-01-01T00:00:00Z"
            }
    
    # FastAPI Application
    app = FastAPI(title="Production RAG API", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Security
    security = HTTPBearer()
    
    # Global RAG system
    rag_system = ProductionRAGSystem(ProductionRAGConfig())
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
    
    @app.post("/query")
    @limiter.limit("100/minute")
    async def query_rag(
        request: Request,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """Query the RAG system"""
        # Verify API key
        if not rag_system.security_manager.verify_api_key(credentials):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        try:
            response = await rag_system.ask_question(
                question=question,
                user_id=user_id,
                session_id=session_id
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        from prometheus_client import generate_latest
        return generate_latest()
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize system on startup"""
        logger.info("RAG system starting up")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("RAG system shutting down")
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

## 🚀 Deployment & DevOps

### 🐳 Docker Containerization

=== "📦 Dockerfile"

    ```dockerfile
    FROM python:3.11-slim
    
    # Set working directory
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements first for better caching
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy application code
    COPY . .
    
    # Create non-root user
    RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
    USER raguser
    
    # Expose port
    EXPOSE 8000
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1
    
    # Start command
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

=== "🔧 docker-compose.yml"

    ```yaml
    version: '3.8'
    
    services:
      rag-api:
        build: .
        ports:
          - "8000:8000"
        environment:
          - REDIS_URL=redis://redis:6379
          - OPENAI_API_KEY=${OPENAI_API_KEY}
        depends_on:
          - redis
          - chroma
        volumes:
          - ./logs:/app/logs
          - ./data:/app/data
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
    
      redis:
        image: redis:7-alpine
        ports:
          - "6379:6379"
        volumes:
          - redis_data:/data
    
      chroma:
        image: chromadb/chroma:latest
        ports:
          - "8001:8000"
        volumes:
          - chroma_data:/chroma/chroma
    
      prometheus:
        image: prom/prometheus:latest
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml
          - prometheus_data:/prometheus
    
      grafana:
        image: grafana/grafana:latest
        ports:
          - "3000:3000"
        environment:
          - GF_SECURITY_ADMIN_PASSWORD=admin
        volumes:
          - grafana_data:/var/lib/grafana
    
    volumes:
      redis_data:
      chroma_data:
      prometheus_data:
      grafana_data:
    ```

### ☸️ Kubernetes Deployment

=== "🗂️ k8s-deployment.yaml"

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: rag-api
      labels:
        app: rag-api
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: rag-api
      template:
        metadata:
          labels:
            app: rag-api
        spec:
          containers:
          - name: rag-api
            image: your-registry/rag-api:latest
            ports:
            - containerPort: 8000
            env:
            - name: REDIS_URL
              value: "redis://redis-service:6379"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key
            resources:
              requests:
                memory: "512Mi"
                cpu: "500m"
              limits:
                memory: "1Gi"
                cpu: "1000m"
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 5
              periodSeconds: 5
    
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: rag-api-service
    spec:
      selector:
        app: rag-api
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
      type: LoadBalancer
    ```

## 📊 Monitoring & Observability

### 📈 Metrics Collection

=== "📊 Custom Metrics"

    ```python
    from prometheus_client import Counter, Histogram, Gauge, Summary
    
    # Define metrics
    QUERY_COUNTER = Counter(
        'rag_queries_total',
        'Total number of RAG queries',
        ['status', 'user_type']
    )
    
    QUERY_DURATION = Histogram(
        'rag_query_duration_seconds',
        'Time spent processing queries',
        ['query_type']
    )
    
    RETRIEVAL_ACCURACY = Gauge(
        'rag_retrieval_accuracy',
        'Current retrieval accuracy score'
    )
    
    CACHE_HIT_RATE = Summary(
        'rag_cache_hit_rate',
        'Cache hit rate percentage'
    )
    
    class MetricsCollector:
        """Collect and expose RAG metrics"""
        
        def __init__(self):
            self.query_counter = QUERY_COUNTER
            self.query_duration = QUERY_DURATION
            self.retrieval_accuracy = RETRIEVAL_ACCURACY
            self.cache_hit_rate = CACHE_HIT_RATE
        
        def record_query(self, status: str, user_type: str = "regular"):
            """Record query with status"""
            self.query_counter.labels(status=status, user_type=user_type).inc()
        
        def record_query_duration(self, duration: float, query_type: str = "standard"):
            """Record query processing time"""
            self.query_duration.labels(query_type=query_type).observe(duration)
        
        def update_retrieval_accuracy(self, accuracy: float):
            """Update current retrieval accuracy"""
            self.retrieval_accuracy.set(accuracy)
        
        def record_cache_hit(self, hit_rate: float):
            """Record cache hit rate"""
            self.cache_hit_rate.observe(hit_rate)
    ```

### 🎯 Evaluation Framework

=== "📊 Comprehensive Evaluation"

    ```python
    from typing import List, Dict, Any
    from dataclasses import dataclass
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    
    @dataclass
    class EvaluationResult:
        """Evaluation result container"""
        precision: float
        recall: float
        f1_score: float
        accuracy: float
        latency: float
        cost: float
        
    class RAGEvaluator:
        """Comprehensive RAG evaluation framework"""
        
        def __init__(self, ground_truth_data: List[Dict]):
            self.ground_truth = ground_truth_data
            self.metrics_history = []
        
        def evaluate_retrieval(self, rag_system, test_queries: List[str]) -> Dict:
            """Evaluate retrieval quality"""
            precisions, recalls, f1_scores = [], [], []
            
            for query in test_queries:
                # Get ground truth for this query
                ground_truth_docs = self._get_ground_truth_docs(query)
                
                # Get RAG system results
                retrieved_docs = rag_system.retrieve(query)
                
                # Calculate metrics
                precision, recall, f1 = self._calculate_retrieval_metrics(
                    retrieved_docs, ground_truth_docs
                )
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            return {
                'avg_precision': np.mean(precisions),
                'avg_recall': np.mean(recalls),
                'avg_f1': np.mean(f1_scores),
                'precision_std': np.std(precisions),
                'recall_std': np.std(recalls),
                'f1_std': np.std(f1_scores)
            }
        
        def evaluate_generation(self, rag_system, test_queries: List[str]) -> Dict:
            """Evaluate generation quality"""
            results = []
            
            for query in test_queries:
                # Get ground truth answer
                ground_truth_answer = self._get_ground_truth_answer(query)
                
                # Get RAG system answer
                rag_answer = rag_system.ask_question(query)['answer']
                
                # Calculate similarity metrics
                similarity_score = self._calculate_answer_similarity(
                    rag_answer, ground_truth_answer
                )
                
                results.append({
                    'query': query,
                    'similarity_score': similarity_score,
                    'generated_answer': rag_answer,
                    'ground_truth': ground_truth_answer
                })
            
            return {
                'avg_similarity': np.mean([r['similarity_score'] for r in results]),
                'results': results
            }
        
        def _calculate_retrieval_metrics(self, retrieved_docs, ground_truth_docs):
            """Calculate precision, recall, F1 for retrieval"""
            retrieved_ids = set(doc.metadata.get('id') for doc in retrieved_docs)
            ground_truth_ids = set(doc['id'] for doc in ground_truth_docs)
            
            if not ground_truth_ids:
                return 0.0, 0.0, 0.0
            
            true_positives = len(retrieved_ids & ground_truth_ids)
            false_positives = len(retrieved_ids - ground_truth_ids)
            false_negatives = len(ground_truth_ids - retrieved_ids)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
        
        def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
            """Calculate similarity between two answers"""
            # This is a simplified implementation
            # In practice, you'd use BERT, ROUGE, or other metrics
            from difflib import SequenceMatcher
            return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()
        
        def _get_ground_truth_docs(self, query: str) -> List[Dict]:
            """Get ground truth documents for a query"""
            for item in self.ground_truth:
                if item['query'] == query:
                    return item['relevant_docs']
            return []
        
        def _get_ground_truth_answer(self, query: str) -> str:
            """Get ground truth answer for a query"""
            for item in self.ground_truth:
                if item['query'] == query:
                    return item['answer']
            return ""
    ```

## 🔧 Advanced Tools & Frameworks

### 🦙 LangChain Integration

=== "🔗 LangChain Pipeline"

    ```python
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.callbacks import StdOutCallbackHandler
    
    class LangChainRAGPipeline:
        """Advanced RAG using LangChain"""
        
        def __init__(self, vector_store, llm):
            self.vector_store = vector_store
            self.llm = llm
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=1000,
                memory_key="chat_history",
                return_messages=True
            )
            self.chain = self._create_chain()
        
        def _create_chain(self):
            """Create conversational retrieval chain"""
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory,
                callbacks=[StdOutCallbackHandler()],
                verbose=True
            )
        
        def chat(self, message: str) -> str:
            """Chat with context memory"""
            response = self.chain({"question": message})
            return response["answer"]
    ```

### 🦾 LlamaIndex Integration

=== "🦙 LlamaIndex Advanced"

    ```python
    from llama_index import (
        VectorStoreIndex,
        ServiceContext,
        QueryBundle,
        StorageContext
    )
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.postprocessor import SimilarityPostprocessor
    
    class LlamaIndexRAGPipeline:
        """Advanced RAG using LlamaIndex"""
        
        def __init__(self, documents, service_context):
            self.service_context = service_context
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context
            )
            self.query_engine = self._create_query_engine()
        
        def _create_query_engine(self):
            """Create advanced query engine"""
            # Configure retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10
            )
            
            # Configure postprocessor
            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=0.7
            )
            
            # Create query engine
            return RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[postprocessor],
                service_context=self.service_context
            )
        
        def query(self, question: str) -> str:
            """Query with advanced processing"""
            query_bundle = QueryBundle(question)
            response = self.query_engine.query(query_bundle)
            return response.response
    ```

## 🎯 Best Practices & Tips

### ⚡ Performance Optimization

=== "🏃‍♂️ Speed Optimization"

    ```python
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    class OptimizedRAGSystem:
        """Performance-optimized RAG system"""
        
        def __init__(self, config):
            self.config = config
            self.executor = ThreadPoolExecutor(max_workers=10)
            self.embedding_cache = {}
        
        async def parallel_retrieval(self, query: str) -> List[Dict]:
            """Parallel retrieval from multiple sources"""
            tasks = [
                self.vector_search(query),
                self.keyword_search(query),
                self.semantic_search(query)
            ]
            
            results = await asyncio.gather(*tasks)
            return self._merge_results(results)
        
        def _merge_results(self, results_list):
            """Merge results from different retrievers"""
            merged = []
            for results in results_list:
                merged.extend(results)
            
            # Remove duplicates and rank
            unique_results = self._deduplicate(merged)
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)
        
        def batch_embed(self, texts: List[str]) -> List[List[float]]:
            """Batch embedding for efficiency"""
            # Check cache first
            uncached_texts = []
            cached_embeddings = []
            
            for text in texts:
                if text in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[text])
                else:
                    uncached_texts.append(text)
            
            # Embed uncached texts
            if uncached_texts:
                new_embeddings = self.embedding_model.embed_documents(uncached_texts)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.embedding_cache[text] = embedding
                
                cached_embeddings.extend(new_embeddings)
            
            return cached_embeddings
    ```

### 🔒 Security & Privacy

=== "🛡️ Security Implementation"

    ```python
    import hashlib
    import hmac
    from typing import Optional
    
    class SecureRAGSystem:
        """Security-focused RAG implementation"""
        
        def __init__(self, config):
            self.config = config
            self.api_keys = self._load_api_keys()
            self.user_permissions = self._load_user_permissions()
        
        def authenticate_user(self, api_key: str) -> Optional[str]:
            """Authenticate user and return user ID"""
            user_id = self.api_keys.get(api_key)
            return user_id
        
        def authorize_query(self, user_id: str, query: str) -> bool:
            """Check if user is authorized for this query"""
            permissions = self.user_permissions.get(user_id, {})
            
            # Check query patterns
            forbidden_patterns = permissions.get('forbidden_patterns', [])
            for pattern in forbidden_patterns:
                if pattern.lower() in query.lower():
                    return False
            
            # Check document access
            allowed_sources = permissions.get('allowed_sources', [])
            if allowed_sources:
                # Would check if query accesses only allowed sources
                pass
            
            return True
        
        def sanitize_query(self, query: str) -> str:
            """Sanitize user input"""
            # Remove potential injection attempts
            sanitized = query.replace('<', '&lt;').replace('>', '&gt;')
            
            # Limit length
            if len(sanitized) > 1000:
                sanitized = sanitized[:1000]
            
            return sanitized
        
        def anonymize_response(self, response: str, user_id: str) -> str:
            """Anonymize sensitive information in response"""
            # This would implement your anonymization logic
            return response
    ```

*Ready to implement your own RAG system? Choose the pattern that fits your needs and start building!* 🚀
