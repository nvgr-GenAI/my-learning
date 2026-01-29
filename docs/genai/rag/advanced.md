# Advanced RAG: Next-Level Techniques

!!! tip "🚀 Ready for Advanced RAG?"
    You've mastered the basics, now let's explore cutting-edge techniques that make RAG systems truly powerful. Welcome to the advanced tier!

## 🎯 What You'll Learn

This guide covers advanced RAG techniques that separate good systems from great ones:

- **Query Enhancement**: Sophisticated query rewriting and expansion
- **Multi-Modal RAG**: Working with text, images, and audio
- **Agentic RAG**: RAG systems that can reason and take actions
- **Advanced Retrieval**: Hybrid search, dense retrieval, and re-ranking
- **Contextual Compression**: Smart context management
- **Self-Reflective RAG**: Systems that evaluate and improve themselves

## 🧠 Advanced Query Processing

### 🔄 Query Rewriting & Expansion

=== "🎯 Smart Query Enhancement"

    **The Challenge**: User queries are often ambiguous or incomplete
    
    **The Solution**: Intelligent query enhancement that understands intent
    
    ```python
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    
    class AdvancedQueryProcessor:
        """Advanced query processing with multiple enhancement strategies"""
        
        def __init__(self, llm):
            self.llm = llm
            self.query_rewriter = self._create_query_rewriter()
            self.query_expander = self._create_query_expander()
        
        def _create_query_rewriter(self):
            """Create query rewriting chain"""
            template = """
            You are a query optimization expert. Rewrite the user's query to be more 
            specific and effective for document retrieval.
            
            Original Query: {query}
            Context: {context}
            
            Consider:
            1. Add missing context or clarification
            2. Expand abbreviations
            3. Add relevant synonyms
            4. Make implicit information explicit
            
            Rewritten Query:
            """
            return PromptTemplate(template=template, input_variables=["query", "context"])
        
        def enhance_query(self, query: str, context: str = "") -> dict:
            """Enhance query with multiple strategies"""
            # 1. Rewrite for clarity
            rewritten = self._rewrite_query(query, context)
            
            # 2. Generate variations
            variations = self._generate_variations(query)
            
            # 3. Extract entities and concepts
            entities = self._extract_entities(query)
            concepts = self._extract_concepts(query)
            
            # 4. Create semantic expansions
            expansions = self._semantic_expansion(query)
            
            return {
                "original": query,
                "rewritten": rewritten,
                "variations": variations,
                "entities": entities,
                "concepts": concepts,
                "expansions": expansions
            }
        
        def _generate_variations(self, query: str) -> list:
            """Generate multiple query variations"""
            template = """
            Generate 3 different ways to ask the same question:
            
            Original: {query}
            
            Variations:
            1. [More specific version]
            2. [Alternative phrasing]
            3. [Broader context version]
            """
            
            response = self.llm(template.format(query=query))
            # Parse response to extract variations
            return self._parse_variations(response)
    ```

=== "🔍 Multi-Strategy Retrieval"

    **Combining Multiple Retrieval Methods**
    
    ```python
    from typing import List, Dict, Any
    import asyncio
    
    class HybridRetriever:
        """Advanced retrieval combining multiple strategies"""
        
        def __init__(self, vector_store, bm25_retriever, graph_retriever):
            self.vector_store = vector_store
            self.bm25_retriever = bm25_retriever
            self.graph_retriever = graph_retriever
        
        async def retrieve(self, query_bundle: dict, top_k: int = 10) -> List[Dict]:
            """Retrieve using multiple strategies in parallel"""
            
            # Prepare different query formats
            original_query = query_bundle["original"]
            rewritten_query = query_bundle["rewritten"]
            variations = query_bundle["variations"]
            
            # Run retrieval strategies in parallel
            retrieval_tasks = [
                self._vector_retrieval(original_query, top_k),
                self._vector_retrieval(rewritten_query, top_k),
                self._keyword_retrieval(original_query, top_k),
                self._graph_retrieval(query_bundle["entities"], top_k),
                self._semantic_retrieval(query_bundle["concepts"], top_k)
            ]
            
            # Add variation retrievals
            for variation in variations:
                retrieval_tasks.append(
                    self._vector_retrieval(variation, top_k // 2)
                )
            
            # Execute all retrievals
            results = await asyncio.gather(*retrieval_tasks)
            
            # Combine and rank results
            combined_results = self._combine_results(results, query_bundle)
            
            return combined_results[:top_k]
        
        def _combine_results(self, results: List[List], query_bundle: dict) -> List[Dict]:
            """Intelligently combine results from different strategies"""
            
            # Collect all unique documents
            all_docs = {}
            
            for strategy_idx, strategy_results in enumerate(results):
                strategy_weight = self._get_strategy_weight(strategy_idx)
                
                for doc in strategy_results:
                    doc_id = doc.metadata.get('id', hash(doc.page_content))
                    
                    if doc_id not in all_docs:
                        all_docs[doc_id] = {
                            'document': doc,
                            'scores': {},
                            'combined_score': 0
                        }
                    
                    # Add strategy score
                    all_docs[doc_id]['scores'][f'strategy_{strategy_idx}'] = doc.metadata.get('score', 0)
                    all_docs[doc_id]['combined_score'] += doc.metadata.get('score', 0) * strategy_weight
            
            # Sort by combined score
            ranked_docs = sorted(
                all_docs.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )
            
            return [doc_info['document'] for doc_info in ranked_docs]
        
        def _get_strategy_weight(self, strategy_idx: int) -> float:
            """Get weight for different retrieval strategies"""
            weights = {
                0: 1.0,  # Original vector search
                1: 0.9,  # Rewritten vector search
                2: 0.7,  # Keyword search
                3: 0.6,  # Graph search
                4: 0.5,  # Semantic search
            }
            return weights.get(strategy_idx, 0.3)  # Variations get lower weight
    ```

### 🧩 Contextual Compression

=== "🗜️ Smart Context Management"

    **Problem**: Too much context overwhelms the LLM
    **Solution**: Intelligent context compression and filtering
    
    ```python
    from langchain.document_transformers import (
        EmbeddingsRedundantFilter,
        EmbeddingsClusteringFilter,
        DocumentCompressorPipeline
    )
    
    class ContextualCompressor:
        """Advanced context compression and filtering"""
        
        def __init__(self, embeddings_model):
            self.embeddings = embeddings_model
            self.compressor = self._create_compressor()
        
        def _create_compressor(self):
            """Create compression pipeline"""
            # Remove redundant documents
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.76
            )
            
            # Cluster similar documents
            clustering_filter = EmbeddingsClusteringFilter(
                embeddings=self.embeddings,
                num_clusters=5,
                sorted=True
            )
            
            # Create pipeline
            return DocumentCompressorPipeline(
                transformers=[redundant_filter, clustering_filter]
            )
        
        def compress_context(self, documents: List, query: str) -> List:
            """Compress context while preserving relevance"""
            
            # Step 1: Remove redundant documents
            filtered_docs = self.compressor.compress_documents(documents, query)
            
            # Step 2: Summarize long documents
            compressed_docs = []
            for doc in filtered_docs:
                if len(doc.page_content) > 1000:  # If document is long
                    summary = self._summarize_document(doc, query)
                    doc.page_content = summary
                
                compressed_docs.append(doc)
            
            # Step 3: Rerank by relevance
            reranked_docs = self._rerank_documents(compressed_docs, query)
            
            return reranked_docs
        
        def _summarize_document(self, doc, query):
            """Summarize document focusing on query relevance"""
            # This would use a summarization model
            # For now, return truncated version
            return doc.page_content[:500] + "..."
    ```

## 🎨 Multi-Modal RAG

### 🖼️ Text + Image RAG

=== "📸 Vision-Language RAG"

    **Capability**: Process both text and images in your knowledge base
    
    ```python
    from langchain.document_loaders import ImageCaptionLoader
    from langchain.embeddings import OpenAIEmbeddings
    from PIL import Image
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    class MultiModalRAG:
        """RAG system that handles text and images"""
        
        def __init__(self):
            self.text_embeddings = OpenAIEmbeddings()
            self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vector_store = None
        
        def process_multimodal_documents(self, docs_with_images):
            """Process documents containing both text and images"""
            processed_docs = []
            
            for doc in docs_with_images:
                # Process text content
                text_content = doc.get('text', '')
                
                # Process images
                image_descriptions = []
                for image_path in doc.get('images', []):
                    description = self._describe_image(image_path)
                    image_descriptions.append(description)
                
                # Combine text and image descriptions
                combined_content = f"{text_content}\n\nImages described: {' '.join(image_descriptions)}"
                
                processed_docs.append({
                    'content': combined_content,
                    'metadata': doc.get('metadata', {}),
                    'images': doc.get('images', [])
                })
            
            return processed_docs
        
        def _describe_image(self, image_path: str) -> str:
            """Generate description for an image"""
            image = Image.open(image_path)
            inputs = self.image_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.image_model.generate(**inputs, max_length=50)
                description = self.image_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return description
        
        def query_multimodal(self, query: str, include_images: bool = False):
            """Query both text and image content"""
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=5)
            
            # Prepare context
            context_parts = []
            image_paths = []
            
            for doc in relevant_docs:
                context_parts.append(doc.page_content)
                if include_images and 'images' in doc.metadata:
                    image_paths.extend(doc.metadata['images'])
            
            # Generate response
            response = self._generate_multimodal_response(
                query, context_parts, image_paths
            )
            
            return response
    ```

### 🎵 Audio RAG

=== "🎙️ Audio-Enhanced RAG"

    **Capability**: Process audio content (podcasts, recordings, etc.)
    
    ```python
    import whisper
    from pydub import AudioSegment
    
    class AudioRAG:
        """RAG system that can process audio content"""
        
        def __init__(self):
            self.whisper_model = whisper.load_model("base")
            self.text_rag = None  # Your existing text RAG system
        
        def process_audio_documents(self, audio_files):
            """Convert audio files to text and add to knowledge base"""
            transcribed_docs = []
            
            for audio_file in audio_files:
                # Transcribe audio
                transcript = self._transcribe_audio(audio_file)
                
                # Create document
                doc = {
                    'content': transcript,
                    'metadata': {
                        'source': audio_file,
                        'type': 'audio_transcript',
                        'duration': self._get_audio_duration(audio_file)
                    }
                }
                
                transcribed_docs.append(doc)
            
            return transcribed_docs
        
        def _transcribe_audio(self, audio_file: str) -> str:
            """Transcribe audio file to text"""
            result = self.whisper_model.transcribe(audio_file)
            return result['text']
        
        def _get_audio_duration(self, audio_file: str) -> float:
            """Get audio file duration in seconds"""
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # Convert to seconds
    ```

## 🤖 Agentic RAG

### 🧠 RAG with Reasoning

=== "🔍 Self-Reflective RAG"

    **Capability**: RAG systems that can evaluate and improve their own responses
    
    ```python
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    
    class AgenticRAG:
        """RAG system with reasoning and self-reflection capabilities"""
        
        def __init__(self, rag_system, llm):
            self.rag_system = rag_system
            self.llm = llm
            self.agent = self._create_agent()
        
        def _create_agent(self):
            """Create agent with RAG tools"""
            
            # Define tools
            tools = [
                Tool(
                    name="Search_Documents",
                    func=self._search_documents,
                    description="Search the knowledge base for relevant information"
                ),
                Tool(
                    name="Evaluate_Answer",
                    func=self._evaluate_answer,
                    description="Evaluate if an answer is complete and accurate"
                ),
                Tool(
                    name="Refine_Query",
                    func=self._refine_query,
                    description="Refine a query to get better search results"
                ),
                Tool(
                    name="Verify_Facts",
                    func=self._verify_facts,
                    description="Verify factual claims in an answer"
                )
            ]
            
            # Create agent
            prompt = PromptTemplate(
                template="""
                You are a research assistant with access to tools. Answer questions
                thoroughly and accurately by using the available tools.
                
                Question: {input}
                
                Use this format:
                Thought: I need to find information about...
                Action: Search_Documents
                Action Input: [query]
                Observation: [results]
                Thought: Let me evaluate this answer...
                Action: Evaluate_Answer
                Action Input: [answer]
                Final Answer: [final response]
                
                Begin!
                {agent_scratchpad}
                """,
                input_variables=["input", "agent_scratchpad"]
            )
            
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            return AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        def _search_documents(self, query: str) -> str:
            """Search documents and return results"""
            results = self.rag_system.retrieve(query)
            return "\n".join([doc.page_content for doc in results])
        
        def _evaluate_answer(self, answer: str) -> str:
            """Evaluate answer quality and completeness"""
            evaluation_prompt = f"""
            Evaluate this answer for:
            1. Completeness (does it fully answer the question?)
            2. Accuracy (are the facts correct?)
            3. Clarity (is it easy to understand?)
            4. Sources (are sources cited?)
            
            Answer: {answer}
            
            Evaluation:
            """
            
            return self.llm(evaluation_prompt)
        
        def _refine_query(self, original_query: str) -> str:
            """Refine query for better results"""
            refinement_prompt = f"""
            The original query might not be getting the best results.
            Suggest a refined version that would be more effective.
            
            Original: {original_query}
            
            Refined query:
            """
            
            return self.llm(refinement_prompt)
        
        def _verify_facts(self, claims: str) -> str:
            """Verify factual claims"""
            # This would integrate with fact-checking services
            # For now, return a placeholder
            return f"Fact-checking results for: {claims}"
        
        def ask_question(self, question: str) -> str:
            """Ask question with agentic reasoning"""
            return self.agent.run(question)
    ```

### 🔄 Iterative RAG

=== "🔄 Self-Improving RAG"

    **Capability**: RAG systems that learn and improve over time
    
    ```python
    from typing import List, Dict, Any
    import json
    from datetime import datetime
    
    class IterativeRAG:
        """RAG system that learns and improves over time"""
        
        def __init__(self, base_rag_system):
            self.base_rag = base_rag_system
            self.query_history = []
            self.feedback_data = []
            self.improvement_log = []
        
        def enhanced_query(self, question: str, max_iterations: int = 3) -> Dict[str, Any]:
            """Query with iterative improvement"""
            
            iteration_results = []
            current_query = question
            
            for i in range(max_iterations):
                # Get initial answer
                result = self.base_rag.ask_question(current_query)
                
                # Evaluate quality
                quality_score = self._evaluate_quality(result, question)
                
                iteration_results.append({
                    'iteration': i + 1,
                    'query': current_query,
                    'answer': result['answer'],
                    'quality_score': quality_score,
                    'sources': result.get('sources', [])
                })
                
                # If quality is high enough, stop
                if quality_score > 0.8:
                    break
                
                # Otherwise, refine the query
                current_query = self._refine_query_based_on_result(
                    question, current_query, result
                )
            
            # Log the interaction
            self._log_interaction(question, iteration_results)
            
            # Return best result
            best_result = max(iteration_results, key=lambda x: x['quality_score'])
            return best_result
        
        def _evaluate_quality(self, result: Dict, original_question: str) -> float:
            """Evaluate the quality of a RAG response"""
            
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            
            # Simple quality metrics
            completeness = len(answer) / 500  # Longer answers might be more complete
            source_quality = len(sources) / 5   # More sources might be better
            
            # Use LLM to evaluate relevance
            relevance_prompt = f"""
            Rate the relevance of this answer to the question on a scale of 0-1:
            
            Question: {original_question}
            Answer: {answer}
            
            Relevance score (0-1):
            """
            
            try:
                relevance_response = self.base_rag.llm(relevance_prompt)
                relevance = float(relevance_response.strip())
            except:
                relevance = 0.5  # Default if parsing fails
            
            # Combine metrics
            quality_score = (completeness + source_quality + relevance) / 3
            return min(quality_score, 1.0)  # Cap at 1.0
        
        def _refine_query_based_on_result(self, original_question: str, current_query: str, result: Dict) -> str:
            """Refine query based on previous result"""
            
            refinement_prompt = f"""
            The previous query didn't get optimal results. Suggest a better query.
            
            Original Question: {original_question}
            Previous Query: {current_query}
            Previous Answer: {result.get('answer', '')}
            
            What information seems to be missing or unclear?
            Suggest a refined query that would get better results:
            """
            
            refined_query = self.base_rag.llm(refinement_prompt)
            return refined_query.strip()
        
        def learn_from_feedback(self, question: str, answer: str, feedback: Dict):
            """Learn from user feedback"""
            
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'feedback': feedback,
                'helpful': feedback.get('helpful', False),
                'accuracy': feedback.get('accuracy', 0),
                'completeness': feedback.get('completeness', 0)
            }
            
            self.feedback_data.append(feedback_entry)
            
            # Analyze feedback patterns
            self._analyze_feedback_patterns()
        
        def _analyze_feedback_patterns(self):
            """Analyze feedback to identify improvement opportunities"""
            
            if len(self.feedback_data) < 10:
                return  # Need more data
            
            # Find common issues
            low_accuracy_queries = [
                f for f in self.feedback_data 
                if f['accuracy'] < 0.7
            ]
            
            low_completeness_queries = [
                f for f in self.feedback_data 
                if f['completeness'] < 0.7
            ]
            
            # Log improvement opportunities
            improvement_entry = {
                'timestamp': datetime.now().isoformat(),
                'total_feedback': len(self.feedback_data),
                'low_accuracy_count': len(low_accuracy_queries),
                'low_completeness_count': len(low_completeness_queries),
                'common_issues': self._identify_common_issues()
            }
            
            self.improvement_log.append(improvement_entry)
        
        def _identify_common_issues(self) -> List[str]:
            """Identify common issues from feedback"""
            # This would analyze feedback text to identify patterns
            # For now, return placeholder
            return ["Need more recent sources", "Answers too technical", "Missing context"]
    ```

## 🎯 Advanced Evaluation Techniques

### 📊 Comprehensive Evaluation Framework

=== "🔬 Advanced Metrics"

    ```python
    from typing import List, Dict, Any
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
    
    class AdvancedRAGEvaluator:
        """Comprehensive evaluation framework for RAG systems"""
        
        def __init__(self):
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.evaluation_history = []
        
        def comprehensive_evaluation(self, 
                                   rag_system, 
                                   test_cases: List[Dict],
                                   include_llm_evaluation: bool = True) -> Dict:
            """Run comprehensive evaluation"""
            
            results = {
                'retrieval_metrics': {},
                'generation_metrics': {},
                'semantic_metrics': {},
                'efficiency_metrics': {},
                'user_experience_metrics': {}
            }
            
            # Evaluate each test case
            for test_case in test_cases:
                case_results = self._evaluate_single_case(
                    rag_system, test_case, include_llm_evaluation
                )
                
                # Aggregate results
                for metric_type, metrics in case_results.items():
                    if metric_type not in results:
                        results[metric_type] = {}
                    
                    for metric_name, value in metrics.items():
                        if metric_name not in results[metric_type]:
                            results[metric_type][metric_name] = []
                        results[metric_type][metric_name].append(value)
            
            # Calculate averages
            for metric_type in results:
                for metric_name in results[metric_type]:
                    values = results[metric_type][metric_name]
                    results[metric_type][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
            
            return results
        
        def _evaluate_single_case(self, rag_system, test_case: Dict, include_llm: bool) -> Dict:
            """Evaluate a single test case"""
            
            query = test_case['query']
            ground_truth = test_case.get('ground_truth', {})
            
            # Get RAG response
            import time
            start_time = time.time()
            response = rag_system.ask_question(query)
            end_time = time.time()
            
            # Calculate metrics
            results = {}
            
            # Retrieval metrics
            if 'relevant_docs' in ground_truth:
                results['retrieval_metrics'] = self._calculate_retrieval_metrics(
                    response.get('sources', []), 
                    ground_truth['relevant_docs']
                )
            
            # Generation metrics
            if 'expected_answer' in ground_truth:
                results['generation_metrics'] = self._calculate_generation_metrics(
                    response.get('answer', ''),
                    ground_truth['expected_answer']
                )
            
            # Semantic metrics
            results['semantic_metrics'] = self._calculate_semantic_metrics(
                response.get('answer', ''),
                ground_truth.get('expected_answer', '')
            )
            
            # Efficiency metrics
            results['efficiency_metrics'] = {
                'response_time': end_time - start_time,
                'tokens_used': len(response.get('answer', '').split()),
                'sources_retrieved': len(response.get('sources', []))
            }
            
            # LLM-based evaluation
            if include_llm:
                results['llm_evaluation'] = self._llm_based_evaluation(
                    query, response.get('answer', ''), ground_truth
                )
            
            return results
        
        def _calculate_semantic_metrics(self, generated_answer: str, expected_answer: str) -> Dict:
            """Calculate semantic similarity metrics"""
            
            if not expected_answer:
                return {}
            
            # Generate embeddings
            gen_embedding = self.sentence_model.encode([generated_answer])
            exp_embedding = self.sentence_model.encode([expected_answer])
            
            # Calculate cosine similarity
            similarity = 1 - cosine(gen_embedding[0], exp_embedding[0])
            
            return {
                'semantic_similarity': similarity,
                'embedding_distance': cosine(gen_embedding[0], exp_embedding[0])
            }
        
        def _llm_based_evaluation(self, query: str, answer: str, ground_truth: Dict) -> Dict:
            """Use LLM to evaluate answer quality"""
            
            evaluation_prompt = f"""
            Evaluate this RAG system response on a scale of 1-10 for each criterion:
            
            Query: {query}
            Answer: {answer}
            Expected Answer: {ground_truth.get('expected_answer', 'Not provided')}
            
            Criteria:
            1. Relevance: How well does the answer address the query?
            2. Accuracy: How factually correct is the answer?
            3. Completeness: How comprehensive is the answer?
            4. Clarity: How clear and understandable is the answer?
            5. Coherence: How well-structured and logical is the answer?
            
            Provide scores as JSON:
            {{
                "relevance": <score>,
                "accuracy": <score>,
                "completeness": <score>,
                "clarity": <score>,
                "coherence": <score>,
                "overall": <score>,
                "explanation": "<brief explanation>"
            }}
            """
            
            # This would use an LLM to evaluate
            # For now, return placeholder scores
            return {
                'relevance': 8.0,
                'accuracy': 7.5,
                'completeness': 7.0,
                'clarity': 8.5,
                'coherence': 8.0,
                'overall': 7.8,
                'explanation': "Good answer with minor issues"
            }
    ```

## 🚀 Production-Ready Advanced RAG

### 🏭 Enterprise Advanced RAG

=== "🛡️ Production Advanced RAG"

    ```python
    from typing import List, Dict, Any, Optional
    import asyncio
    from dataclasses import dataclass
    from contextlib import asynccontextmanager
    
    @dataclass
    class AdvancedRAGConfig:
        """Configuration for advanced RAG system"""
        # Query processing
        enable_query_rewriting: bool = True
        enable_query_expansion: bool = True
        max_query_variations: int = 3
        
        # Retrieval
        enable_hybrid_search: bool = True
        enable_reranking: bool = True
        retrieval_strategies: List[str] = None
        
        # Generation
        enable_iterative_refinement: bool = True
        max_iterations: int = 3
        quality_threshold: float = 0.8
        
        # Multi-modal
        enable_multimodal: bool = False
        supported_modalities: List[str] = None
        
        # Evaluation
        enable_self_evaluation: bool = True
        evaluation_metrics: List[str] = None
        
        def __post_init__(self):
            if self.retrieval_strategies is None:
                self.retrieval_strategies = ['vector', 'keyword', 'semantic']
            if self.supported_modalities is None:
                self.supported_modalities = ['text', 'image']
            if self.evaluation_metrics is None:
                self.evaluation_metrics = ['relevance', 'accuracy', 'completeness']
    
    class ProductionAdvancedRAG:
        """Production-ready advanced RAG system"""
        
        def __init__(self, config: AdvancedRAGConfig):
            self.config = config
            self.query_processor = None
            self.hybrid_retriever = None
            self.context_compressor = None
            self.iterative_generator = None
            self.evaluator = None
            
            # Initialize components
            self._initialize_components()
        
        def _initialize_components(self):
            """Initialize all RAG components"""
            
            if self.config.enable_query_rewriting:
                self.query_processor = AdvancedQueryProcessor(llm=None)  # Initialize with your LLM
            
            if self.config.enable_hybrid_search:
                self.hybrid_retriever = HybridRetriever(
                    vector_store=None,  # Your vector store
                    bm25_retriever=None,  # Your BM25 retriever
                    graph_retriever=None  # Your graph retriever
                )
            
            if self.config.enable_reranking:
                self.context_compressor = ContextualCompressor(
                    embeddings_model=None  # Your embeddings model
                )
            
            if self.config.enable_iterative_refinement:
                self.iterative_generator = IterativeRAG(
                    base_rag_system=None  # Your base RAG system
                )
            
            if self.config.enable_self_evaluation:
                self.evaluator = AdvancedRAGEvaluator()
        
        async def ask_question(self, 
                              question: str, 
                              context: Optional[str] = None,
                              user_id: Optional[str] = None) -> Dict[str, Any]:
            """Advanced question answering with all features"""
            
            # Step 1: Query processing
            if self.config.enable_query_rewriting:
                query_bundle = self.query_processor.enhance_query(question, context or "")
            else:
                query_bundle = {"original": question}
            
            # Step 2: Advanced retrieval
            if self.config.enable_hybrid_search:
                retrieved_docs = await self.hybrid_retriever.retrieve(
                    query_bundle, 
                    top_k=10
                )
            else:
                # Fallback to simple retrieval
                retrieved_docs = self._simple_retrieve(question)
            
            # Step 3: Context compression
            if self.config.enable_reranking:
                compressed_context = self.context_compressor.compress_context(
                    retrieved_docs, 
                    question
                )
            else:
                compressed_context = retrieved_docs[:5]  # Simple truncation
            
            # Step 4: Iterative generation
            if self.config.enable_iterative_refinement:
                response = self.iterative_generator.enhanced_query(
                    question,
                    max_iterations=self.config.max_iterations
                )
            else:
                # Simple generation
                response = self._simple_generate(question, compressed_context)
            
            # Step 5: Self-evaluation
            if self.config.enable_self_evaluation:
                evaluation = self.evaluator._evaluate_single_case(
                    self, 
                    {'query': question}, 
                    include_llm=True
                )
                response['evaluation'] = evaluation
            
            # Step 6: Add metadata
            response['query_processing'] = query_bundle
            response['sources'] = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': getattr(doc, 'score', 0)
                }
                for doc in compressed_context
            ]
            response['config'] = {
                'query_rewriting': self.config.enable_query_rewriting,
                'hybrid_search': self.config.enable_hybrid_search,
                'reranking': self.config.enable_reranking,
                'iterative_refinement': self.config.enable_iterative_refinement,
                'self_evaluation': self.config.enable_self_evaluation
            }
            
            return response
        
        def _simple_retrieve(self, question: str) -> List:
            """Fallback simple retrieval"""
            # Implement basic retrieval
            return []
        
        def _simple_generate(self, question: str, context: List) -> Dict:
            """Fallback simple generation"""
            # Implement basic generation
            return {'answer': 'Simple answer', 'sources': []}
    ```

## 🎓 Next Steps

Ready to implement advanced RAG techniques? Here's your roadmap:

### 🛠️ Implementation Priority

1. **Start with Query Enhancement** - Biggest impact for effort
2. **Add Hybrid Retrieval** - Improves recall and precision
3. **Implement Reranking** - Better relevance scoring
4. **Enable Self-Evaluation** - Quality monitoring
5. **Add Multi-Modal** - Expand capabilities
6. **Build Agentic Features** - Advanced reasoning

### 📚 Recommended Reading

- **Query Enhancement**: Research on query rewriting and expansion
- **Multi-Modal**: CLIP and BLIP model papers
- **Agentic RAG**: ReAct and tool-use papers
- **Evaluation**: RAG evaluation benchmarks and metrics

### 🔧 Tools to Explore

- **Advanced Frameworks**: LangGraph, AutoGen
- **Specialized Models**: Query rewriting models, reranking models
- **Evaluation Tools**: RAGAs, TruLens
- **Multi-Modal**: CLIP, BLIP, Whisper

*Ready to build the next generation of RAG systems? The advanced techniques are waiting for you!* 🚀
