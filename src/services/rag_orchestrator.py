"""
RAG Orchestrator - Coordinates all RAG services
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .interfaces import (
    QueryServiceInterface,
    RetrievalServiceInterface,
    GenerationServiceInterface,
    MemoryServiceInterface,
    MonitoringServiceInterface,
    MemoryContext
)
from ..experimentation.ab_testing_framework import ABTestingFramework
from ..evaluation.ragas_metrics import RAGASEvaluator

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Container for RAG system response"""
    
    answer: str
    contexts: List[str]
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    latency_ms: float
    tokens_generated: int
    
    # Optional evaluation scores
    ragas_scores: Optional[Dict[str, float]] = None
    
    # Query analysis
    query_intent: Optional[str] = None
    query_complexity: Optional[float] = None
    query_entities: List[str] = None
    
    # Retrieval details
    retrieval_method: Optional[str] = None
    contexts_retrieved: int = 0
    contexts_compressed: bool = False
    compression_ratio: Optional[float] = None
    
    # Knowledge graph
    graph_entities_found: int = 0
    graph_relations_used: int = 0

    def __post_init__(self):
        if self.query_entities is None:
            self.query_entities = []

class RAGOrchestrator:
    """
    Orchestrates all RAG services using dependency injection
    Implements single responsibility principle for coordination
    """
    
    def __init__(
        self,
        query_service: QueryServiceInterface,
        retrieval_service: RetrievalServiceInterface,
        generation_service: GenerationServiceInterface,
        memory_service: Optional[MemoryServiceInterface] = None,
        monitoring_service: Optional[MonitoringServiceInterface] = None,
        enable_ab_testing: bool = False,
        enable_evaluation: bool = False
    ):
        """
        Initialize RAG orchestrator with injected services
        
        Args:
            query_service: Service for query optimization
            retrieval_service: Service for document retrieval
            generation_service: Service for response generation
            memory_service: Optional service for conversation memory
            monitoring_service: Optional service for monitoring
            enable_ab_testing: Whether to enable A/B testing
            enable_evaluation: Whether to enable RAGAS evaluation
        """
        self.query_service = query_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
        
        # Optional components
        self.ab_testing = ABTestingFramework() if enable_ab_testing else None
        self.evaluator = RAGASEvaluator() if enable_evaluation else None
        
        self.enable_ab_testing = enable_ab_testing
        self.enable_evaluation = enable_evaluation
        
        logger.info("RAGOrchestrator initialized with injected services")
    
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        evaluate: bool = False
    ) -> RAGResponse:
        """
        Main query processing method that orchestrates all services
        
        Args:
            query: User query
            session_id: Optional session identifier
            user_id: Optional user identifier
            config: Optional configuration overrides
            evaluate: Whether to run RAGAS evaluation
            
        Returns:
            RAGResponse with complete processing results
        """
        start_time = time.time()
        response_metadata = {}
        
        # Default configuration
        default_config = {
            'enable_query_optimization': True,
            'enable_query_decomposition': True,
            'enable_hyde': True,
            'retrieval_method': 'adaptive',
            'top_k': 10,
            'enable_compression': True,
            'max_context_tokens': 2000,
            'use_reranking': True,
            'use_mmr': True,
            'mmr_lambda': 0.5,
            'max_graph_hops': 2,
            'experiment_id': None
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        config = default_config
        
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # A/B testing variant selection
            variant = None
            if self.enable_ab_testing and config.get('experiment_id'):
                variant = self.ab_testing.get_variant(
                    config['experiment_id'],
                    user_id or "anonymous"
                )
                if variant:
                    response_metadata['ab_variant'] = variant.name
            
            # 1. Query Optimization
            query_result = None
            if config['enable_query_optimization']:
                query_result = self.query_service.optimize_query(
                    query=query,
                    enable_decomposition=config['enable_query_decomposition'],
                    enable_hyde=config['enable_hyde']
                )
                
                optimized_query = query_result.optimized_query
                sub_queries = query_result.sub_queries
                query_entities = query_result.entities
                query_intent = query_result.intent
                query_complexity = query_result.complexity
            else:
                optimized_query = query
                sub_queries = [query]
                query_entities = []
                query_intent = None
                query_complexity = None
            
            # 2. Memory Context Retrieval
            conversation_contexts = []
            if self.memory_service:
                # Resolve references first
                optimized_query = self.memory_service.resolve_references(optimized_query)
                
                # Get relevant context
                conversation_contexts = self.memory_service.get_relevant_context(
                    optimized_query, k=3
                )
            
            # 3. Document Retrieval and Context Processing
            retrieval_result = self.retrieval_service.retrieve_contexts(
                query=optimized_query,
                sub_queries=sub_queries,
                entities=query_entities,
                conversation_contexts=conversation_contexts,
                retrieval_method=config['retrieval_method'],
                top_k=config['top_k'],
                enable_compression=config['enable_compression'],
                max_context_tokens=config['max_context_tokens'],
                use_reranking=config['use_reranking'],
                use_mmr=config['use_mmr'],
                mmr_lambda=config['mmr_lambda'],
                max_graph_hops=config['max_graph_hops']
            )
            
            # Track retrieval metrics
            if self.monitoring_service:
                retrieval_time = (time.time() - start_time) * 1000
                self.monitoring_service.track_retrieval(
                    retriever_type=config['retrieval_method'],
                    latency_ms=retrieval_time,
                    contexts_count=len(retrieval_result.contexts)
                )
            
            # 4. Response Generation
            generation_start = time.time()
            
            # Prepare conversation history for generation
            conversation_history = []
            if self.memory_service and conversation_contexts:
                for ctx in conversation_contexts[-3:]:  # Last 3 turns
                    if 'query' in ctx.metadata and 'response' in ctx.metadata:
                        conversation_history.append({
                            'query': ctx.metadata['query'],
                            'response': ctx.metadata['response']
                        })
            
            generation_result = self.generation_service.generate_response(
                query=query,
                contexts=retrieval_result.contexts,
                conversation_history=conversation_history
            )
            
            # Track generation metrics
            if self.monitoring_service:
                self.monitoring_service.track_generation(
                    tokens=generation_result.tokens_generated,
                    latency_ms=generation_result.generation_time_ms,
                    model="llm_model"  # Would get from config
                )
            
            # 5. Update Memory
            if self.memory_service:
                self.memory_service.add_turn(
                    query=query,
                    response=generation_result.answer,
                    contexts=retrieval_result.contexts,
                    relevance_scores=retrieval_result.scores
                )
            
            # 6. RAGAS Evaluation (optional)
            ragas_scores = None
            if evaluate and self.evaluator:
                try:
                    evaluation_result = self.evaluator.evaluate(
                        question=query,
                        answer=generation_result.answer,
                        contexts=retrieval_result.contexts,
                        ground_truth=None
                    )
                    ragas_scores = {
                        'faithfulness': evaluation_result.faithfulness,
                        'answer_relevancy': evaluation_result.answer_relevancy,
                        'context_relevancy': evaluation_result.context_relevancy,
                        'context_precision': evaluation_result.context_precision
                    }
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")
            
            # 7. A/B Testing Metrics
            if variant and self.enable_ab_testing:
                metrics = {
                    'response_quality': sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0,
                    'latency': (time.time() - start_time) * 1000,
                    'tokens': generation_result.tokens_generated
                }
                
                self.ab_testing.record_event(
                    config['experiment_id'],
                    user_id or "anonymous",
                    variant.name,
                    metrics
                )
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            # Track overall request
            if self.monitoring_service:
                self.monitoring_service.track_request(
                    method="query",
                    duration_ms=total_latency,
                    status="success",
                    metadata={
                        'tokens': generation_result.tokens_generated,
                        'contexts': len(retrieval_result.contexts),
                        'intent': query_intent
                    }
                )
            
            # Build final response
            rag_response = RAGResponse(
                answer=generation_result.answer,
                contexts=retrieval_result.contexts,
                scores={'retrieval': retrieval_result.scores} if retrieval_result.scores else {},
                metadata=response_metadata,
                latency_ms=total_latency,
                tokens_generated=generation_result.tokens_generated,
                ragas_scores=ragas_scores,
                query_intent=query_intent,
                query_complexity=query_complexity,
                query_entities=query_entities,
                retrieval_method=config['retrieval_method'],
                contexts_retrieved=len(retrieval_result.contexts),
                contexts_compressed=retrieval_result.contexts_compressed,
                compression_ratio=retrieval_result.compression_ratio,
                graph_entities_found=retrieval_result.graph_entities_found,
                graph_relations_used=retrieval_result.graph_relations_used
            )
            
            logger.info(f"Query processed successfully in {total_latency:.0f}ms")
            return rag_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Track error
            if self.monitoring_service:
                self.monitoring_service.track_error(
                    error_type="query_error",
                    component="rag_orchestrator",
                    message=str(e)
                )
                
                self.monitoring_service.create_alert(
                    name="query_failure",
                    severity="error",
                    message=f"Query processing failed: {str(e)}"
                )
            
            # Return error response
            return RAGResponse(
                answer=f"I apologize, but I encountered an error processing your query: {str(e)}",
                contexts=[],
                scores={},
                metadata={'error': str(e)},
                latency_ms=(time.time() - start_time) * 1000,
                tokens_generated=0
            )
    
    def index_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None
    ) -> None:
        """
        Index a document using the retrieval service
        
        Args:
            content: Document content
            doc_id: Unique document identifier
            metadata: Optional document metadata
            doc_type: Optional document type for optimization
        """
        start_time = time.time()
        logger.info(f"Indexing document: {doc_id}")
        
        try:
            self.retrieval_service.index_document(
                content=content,
                doc_id=doc_id,
                metadata=metadata,
                doc_type=doc_type
            )
            
            # Track indexing
            if self.monitoring_service:
                self.monitoring_service.track_request(
                    method="index_document",
                    duration_ms=(time.time() - start_time) * 1000,
                    status="success",
                    metadata={"doc_id": doc_id}
                )
            
            logger.info(f"Document {doc_id} indexed successfully")
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            
            if self.monitoring_service:
                self.monitoring_service.track_error(
                    error_type="indexing_error",
                    component="rag_orchestrator",
                    message=str(e)
                )
            
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status from all services"""
        status = {
            'timestamp': datetime.now(),
            'services': {}
        }
        
        # Get service-specific statistics
        try:
            if hasattr(self.query_service, 'get_query_statistics'):
                status['services']['query_service'] = self.query_service.get_query_statistics()
        except Exception as e:
            logger.warning(f"Error getting query service stats: {e}")
        
        try:
            if hasattr(self.retrieval_service, 'get_retrieval_statistics'):
                status['services']['retrieval_service'] = self.retrieval_service.get_retrieval_statistics()
        except Exception as e:
            logger.warning(f"Error getting retrieval service stats: {e}")
        
        try:
            if hasattr(self.generation_service, 'get_generation_statistics'):
                status['services']['generation_service'] = self.generation_service.get_generation_statistics()
        except Exception as e:
            logger.warning(f"Error getting generation service stats: {e}")
        
        try:
            if self.memory_service and hasattr(self.memory_service, 'get_memory_statistics'):
                status['services']['memory_service'] = self.memory_service.get_memory_statistics()
        except Exception as e:
            logger.warning(f"Error getting memory service stats: {e}")
        
        try:
            if self.monitoring_service:
                status['services']['monitoring_service'] = self.monitoring_service.get_health_status()
        except Exception as e:
            logger.warning(f"Error getting monitoring service stats: {e}")
        
        return status
    
    def shutdown(self) -> None:
        """Graceful shutdown of all services"""
        logger.info("Shutting down RAG orchestrator...")
        
        # End memory session
        if self.memory_service:
            try:
                self.memory_service.end_session()
                self.memory_service.save_memory()
            except Exception as e:
                logger.warning(f"Error shutting down memory service: {e}")
        
        # Export monitoring metrics
        if self.monitoring_service:
            try:
                self.monitoring_service.export_metrics("./monitoring/final_metrics.json")
            except Exception as e:
                logger.warning(f"Error exporting monitoring metrics: {e}")
        
        logger.info("RAG orchestrator shutdown complete")