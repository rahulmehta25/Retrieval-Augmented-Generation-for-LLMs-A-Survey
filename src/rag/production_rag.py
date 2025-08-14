"""
Production-Grade RAG System with all advanced features
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml
import json
import hashlib
from pathlib import Path
from collections import deque
import numpy as np

# Import our production components
from src.retrieval.hybrid_retriever import HybridRetriever, SmartChunker
from src.optimization.query_optimizer import QueryOptimizer, HyDEGenerator
from src.evaluation.ragas_metrics import RAGASEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGRequest:
    """Production RAG request with metadata"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    experiment_variant: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass 
class RAGResponse:
    """Production RAG response with full telemetry"""
    answer: str
    contexts: List[str]
    scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None
    latency_ms: float = 0
    tokens_used: int = 0
    cache_hit: bool = False
    experiment_variant: Optional[str] = None

class ConversationMemory:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)
        self.context_window = 3
        
    def add_turn(self, query: str, answer: str, contexts: List[str]):
        """Add a conversation turn"""
        self.history.append({
            'query': query,
            'answer': answer,
            'contexts': contexts,
            'timestamp': time.time()
        })
    
    def get_context(self) -> str:
        """Get relevant conversation context"""
        if not self.history:
            return ""
        
        recent = list(self.history)[-self.context_window:]
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"Q: {turn['query']}")
            context_parts.append(f"A: {turn['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def should_use_memory(self, query: str) -> bool:
        """Determine if conversation memory is relevant"""
        # Check for references to previous context
        memory_indicators = ['it', 'that', 'this', 'previous', 'above', 'before', 'mentioned']
        query_lower = query.lower()
        
        return any(indicator in query_lower for indicator in memory_indicators)

class ResponseCache:
    """Simple response cache for common queries"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Create hash key for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[RAGResponse]:
        """Get cached response"""
        key = self._hash_query(query)
        if key in self.cache:
            self.hits += 1
            response = self.cache[key]
            response.cache_hit = True
            return response
        self.misses += 1
        return None
    
    def set(self, query: str, response: RAGResponse):
        """Cache a response"""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        key = self._hash_query(query)
        self.cache[key] = response
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

class ProductionRAG:
    """
    Production-grade RAG system with:
    - Hybrid retrieval
    - Query optimization
    - Conversation memory
    - Response caching
    - A/B testing support
    - Comprehensive monitoring
    """
    
    def __init__(self, config_path: str = "config_production.yaml"):
        """Initialize production RAG system"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self._initialize_components()
        
        # Metrics tracking
        self.metrics = {
            'total_queries': 0,
            'avg_latency': 0,
            'cache_hit_rate': 0,
            'avg_contexts': 0,
            'avg_scores': {}
        }
        
        logger.info("ProductionRAG initialized with advanced features")
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        
        # Chunker
        self.chunker = SmartChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            overlap=self.config['chunking']['chunk_overlap']
        )
        
        # Retriever
        self.retriever = HybridRetriever(
            embedding_model=self.config['embedding']['model'],
            reranker_model=self.config['retrieval']['reranker_model'],
            collection_name=self.config['vector_store']['collection_name'],
            persist_dir=self.config['vector_store']['persist_directory']
        )
        
        # Query optimizer
        self.query_optimizer = QueryOptimizer()
        
        # HyDE generator (if we have LLM)
        self.hyde_generator = None
        
        # Conversation memory
        self.memory = ConversationMemory(
            max_history=self.config['conversation']['max_history']
        )
        
        # Response cache
        self.cache = ResponseCache()
        
        # Evaluator (optional)
        self.evaluator = None
        if self.config['evaluation']['enable_ragas']:
            try:
                from src.generation.generator import Generator
                generator = Generator(self.config['generation'])
                self.evaluator = RAGASEvaluator(llm_generator=generator)
            except:
                logger.warning("RAGAS evaluator not available")
        
        # LLM Generator
        try:
            from src.generation.generator import Generator
            self.generator = Generator(self.config['generation'])
        except:
            logger.error("Generator not available, using mock")
            self.generator = None
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Index documents with smart chunking"""
        
        all_chunks = []
        all_metadata = []
        
        for i, doc in enumerate(documents):
            # Smart chunking
            chunks = self.chunker.chunk_document(
                doc,
                metadata[i] if metadata else {'doc_id': i}
            )
            
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                all_metadata.append(chunk['metadata'])
        
        # Add to retriever
        self.retriever.add_documents(
            all_chunks,
            doc_ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            metadata=all_metadata
        )
        
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    
    def _apply_query_optimization(self, query: str) -> Tuple[str, Dict]:
        """Apply query optimization techniques"""
        
        optimized = self.query_optimizer.optimize(query)
        
        # Use the best version
        final_query = optimized.rewritten or query
        
        # If complex, use first decomposed question
        if optimized.decomposed and len(optimized.decomposed) > 1:
            final_query = optimized.decomposed[0]
        
        optimization_metadata = {
            'intent': optimized.intent,
            'complexity': optimized.complexity,
            'keywords': optimized.keywords,
            'decomposed': optimized.decomposed
        }
        
        return final_query, optimization_metadata
    
    def _retrieve_contexts(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant contexts"""
        
        # Check if conversation memory is relevant
        if self.memory.should_use_memory(query):
            conversation_context = self.memory.get_context()
            if conversation_context:
                query = f"{conversation_context}\n\nCurrent question: {query}"
        
        # Apply HyDE if available
        if self.hyde_generator and self.config['query_optimization']['use_hyde']:
            hypothetical = self.hyde_generator.generate_hypothetical_answer(query)
            # Use hypothetical for retrieval
            contexts = self.retriever.retrieve(
                hypothetical,
                k=k,
                use_reranking=self.config['retrieval']['rerank']
            )
        else:
            # Regular retrieval
            contexts = self.retriever.retrieve(
                query,
                k=k,
                use_reranking=self.config['retrieval']['rerank'],
                alpha=self.config['retrieval']['hybrid_alpha']
            )
        
        return contexts
    
    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer from contexts"""
        
        if not self.generator:
            # Fallback to simple extraction
            return self._extract_answer(query, contexts)
        
        # Build prompt
        context_text = "\n\n".join(contexts)
        
        # Add conversation memory if relevant
        conversation_context = ""
        if self.memory.should_use_memory(query):
            conversation_context = self.memory.get_context()
        
        prompt = f"""Based on the following context, answer the question.
        
{conversation_context}

Context:
{context_text}

Question: {query}
Answer:"""
        
        try:
            answer = self.generator.generate(prompt, "")
            return answer if answer else "I cannot find relevant information to answer this question."
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._extract_answer(query, contexts)
    
    def _extract_answer(self, query: str, contexts: List[str]) -> str:
        """Simple answer extraction fallback"""
        if not contexts:
            return "No relevant information found."
        
        # Return first context as answer
        return contexts[0][:500] + "..." if len(contexts[0]) > 500 else contexts[0]
    
    def _evaluate_response(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Evaluate response quality"""
        
        if not self.evaluator:
            return {}
        
        try:
            scores = self.evaluator.evaluate(
                question=query,
                answer=answer,
                contexts=contexts,
                ground_truth=None
            )
            return scores
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def query(
        self,
        request: RAGRequest,
        use_cache: bool = True,
        evaluate: bool = False
    ) -> RAGResponse:
        """
        Main query method for production RAG
        
        Args:
            request: RAG request with query and metadata
            use_cache: Whether to use response cache
            evaluate: Whether to evaluate response quality
        
        Returns:
            RAGResponse with answer and full telemetry
        """
        
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(request.query)
            if cached:
                logger.info(f"Cache hit for query: {request.query[:50]}...")
                return cached
        
        # Apply query optimization
        optimized_query, opt_metadata = self._apply_query_optimization(request.query)
        
        # Retrieve contexts
        contexts = self._retrieve_contexts(
            optimized_query,
            k=self.config['retrieval']['k_documents']
        )
        
        # Generate answer
        answer = self._generate_answer(optimized_query, contexts)
        
        # Evaluate if requested
        scores = None
        if evaluate:
            scores = self._evaluate_response(request.query, answer, contexts)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = len(answer.split()) + sum(len(c.split()) for c in contexts)
        
        # Create response
        response = RAGResponse(
            answer=answer,
            contexts=contexts[:2],  # Return top 2 for display
            scores=scores,
            metadata={
                'optimization': opt_metadata,
                'retriever_stats': self.retriever.get_statistics(),
                'cache_stats': self.cache.get_stats()
            },
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cache_hit=False,
            experiment_variant=request.experiment_variant
        )
        
        # Update conversation memory
        self.memory.add_turn(request.query, answer, contexts)
        
        # Cache response
        if use_cache:
            self.cache.set(request.query, response)
        
        # Update metrics
        self._update_metrics(response)
        
        logger.info(f"Query processed in {latency_ms:.2f}ms, {len(contexts)} contexts retrieved")
        
        return response
    
    def _update_metrics(self, response: RAGResponse):
        """Update system metrics"""
        self.metrics['total_queries'] += 1
        
        # Update running averages
        n = self.metrics['total_queries']
        self.metrics['avg_latency'] = (
            (self.metrics['avg_latency'] * (n-1) + response.latency_ms) / n
        )
        self.metrics['avg_contexts'] = (
            (self.metrics['avg_contexts'] * (n-1) + len(response.contexts)) / n
        )
        
        if response.scores:
            for metric, score in response.scores.items():
                if metric not in self.metrics['avg_scores']:
                    self.metrics['avg_scores'][metric] = 0
                self.metrics['avg_scores'][metric] = (
                    (self.metrics['avg_scores'][metric] * (n-1) + score) / n
                )
    
    def get_metrics(self) -> Dict:
        """Get system metrics"""
        return {
            **self.metrics,
            'cache_stats': self.cache.get_stats(),
            'retriever_stats': self.retriever.get_statistics()
        }
    
    def health_check(self) -> Dict:
        """System health check"""
        health = {
            'status': 'healthy',
            'components': {
                'retriever': self.retriever is not None,
                'optimizer': self.query_optimizer is not None,
                'generator': self.generator is not None,
                'evaluator': self.evaluator is not None,
                'cache': self.cache is not None,
                'memory': self.memory is not None
            },
            'metrics': self.get_metrics()
        }
        
        # Check if any critical component is down
        if not health['components']['retriever'] or not health['components']['generator']:
            health['status'] = 'degraded'
        
        return health