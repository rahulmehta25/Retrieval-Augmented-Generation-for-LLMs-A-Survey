"""
Fully Integrated Production RAG System
Combines all advanced features into a unified pipeline

This module provides both the new service-oriented architecture and
backward compatibility with the original monolithic implementation.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from datetime import datetime

# Import service-oriented components
from ..services import (
    QueryService,
    RetrievalService, 
    GenerationService,
    MemoryService,
    MonitoringService,
    RAGOrchestrator
)
from ..services.rag_orchestrator import RAGResponse

# Import legacy components for backward compatibility
from ..optimization.semantic_query_optimizer import SemanticQueryOptimizer, QueryRewriter
from ..graph_rag.advanced_knowledge_graph import AdvancedKnowledgeGraph, GraphQuery
from ..retrieval.advanced_hybrid_retriever import AdvancedHybridRetriever
from ..chunking.semantic_chunker import SemanticChunker
from ..retrieval.advanced_context_compressor import AdvancedContextCompressor
from ..memory.advanced_conversation_memory import AdvancedConversationMemory
from ..experimentation.ab_testing_framework import ABTestingFramework
from ..monitoring.production_monitoring import ProductionMonitoring, AlertSeverity
from ..evaluation.ragas_metrics import RAGASEvaluator
from ..generation.generator import LLMGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Configuration for production RAG system"""
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemma:2b"
    
    # Retrieval settings
    retrieval_method: str = "adaptive"  # adaptive, hybrid, dense, sparse
    top_k: int = 10
    use_reranking: bool = True
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    
    # Optimization settings
    enable_query_optimization: bool = True
    enable_query_decomposition: bool = True
    enable_hyde: bool = True
    
    # Knowledge graph settings
    enable_knowledge_graph: bool = True
    max_graph_hops: int = 2
    
    # Memory settings
    enable_conversation_memory: bool = True
    max_conversation_turns: int = 20
    
    # Compression settings
    enable_context_compression: bool = True
    max_context_tokens: int = 2000
    compression_method: str = "adaptive"  # adaptive, query_focused, extractive
    
    # Monitoring settings
    enable_monitoring: bool = True
    prometheus_port: int = 8000
    
    # A/B testing settings
    enable_ab_testing: bool = False
    experiment_id: Optional[str] = None
    
    # Storage paths
    index_path: str = "./rag_index"
    knowledge_graph_path: str = "./knowledge_graph"
    memory_path: str = "./conversation_memory"
    monitoring_path: str = "./monitoring"

# RAGResponse is now imported from services.rag_orchestrator

class ProductionRAGSystem:
    """
    Unified production RAG system with service-oriented architecture
    
    This class now uses dependency injection and service-oriented architecture
    while maintaining backward compatibility with the original API.
    """
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        """Initialize production RAG system with service architecture"""
        
        self.config = config or ProductionConfig()
        logger.info("Initializing production RAG system with service architecture...")
        
        # Initialize individual services
        self._initialize_services()
        
        # Create orchestrator with injected services
        self.orchestrator = RAGOrchestrator(
            query_service=self.query_service,
            retrieval_service=self.retrieval_service,
            generation_service=self.generation_service,
            memory_service=self.memory_service if self.config.enable_conversation_memory else None,
            monitoring_service=self.monitoring_service if self.config.enable_monitoring else None,
            enable_ab_testing=self.config.enable_ab_testing,
            enable_evaluation=True  # Always enable evaluation capability
        )
        
        # Register health checks
        if self.config.enable_monitoring:
            self._register_health_checks()
        
        # Initialize session if memory is enabled
        if self.config.enable_conversation_memory:
            session_id = f"session_{datetime.now().timestamp()}"
            self.memory_service.start_session(session_id)
        
        logger.info("Production RAG system initialized successfully with service architecture")
    
    def _initialize_services(self):
        """Initialize all required services"""
        
        # Query Service
        self.query_service = QueryService()
        
        # Retrieval Service  
        knowledge_graph_path = self.config.knowledge_graph_path if self.config.enable_knowledge_graph else None
        self.retrieval_service = RetrievalService(
            embedding_model=self.config.embedding_model,
            reranker_model=self.config.reranker_model,
            index_path=self.config.index_path,
            knowledge_graph_path=knowledge_graph_path,
            max_context_tokens=self.config.max_context_tokens
        )
        
        # Generation Service
        self.generation_service = GenerationService(
            model_name=self.config.llm_model
        )
        
        # Memory Service (optional)
        if self.config.enable_conversation_memory:
            self.memory_service = MemoryService(
                persist_path=self.config.memory_path
            )
        else:
            self.memory_service = None
        
        # Monitoring Service (optional)
        if self.config.enable_monitoring:
            self.monitoring_service = MonitoringService(
                service_name="production_rag",
                prometheus_port=self.config.prometheus_port,
                persist_path=self.config.monitoring_path,
                enable_prometheus=True
            )
        else:
            self.monitoring_service = None
        
        logger.info("All services initialized successfully")
    
    def _register_health_checks(self):
        """Register component health checks"""
        
        if not self.monitoring_service:
            return
        
        # Query service health check
        def check_query_service():
            try:
                # Test with simple query optimization
                result = self.query_service.optimize_query("test query")
                return True, "Query service operational"
            except Exception as e:
                return False, f"Query service error: {str(e)}"
        
        self.monitoring_service.register_health_check("query_service", check_query_service)
        
        # Retrieval service health check
        def check_retrieval_service():
            try:
                # Test retrieval service statistics
                stats = self.retrieval_service.get_retrieval_statistics()
                return True, "Retrieval service operational"
            except Exception as e:
                return False, f"Retrieval service error: {str(e)}"
        
        self.monitoring_service.register_health_check("retrieval_service", check_retrieval_service)
        
        # Generation service health check
        def check_generation_service():
            try:
                # Test generation service
                stats = self.generation_service.get_generation_statistics()
                return True, "Generation service operational"
            except Exception as e:
                return False, f"Generation service error: {str(e)}"
        
        self.monitoring_service.register_health_check("generation_service", check_generation_service)
        
        # Memory service health check
        if self.memory_service:
            def check_memory_service():
                try:
                    stats = self.memory_service.get_memory_statistics()
                    return True, "Memory service operational"
                except Exception as e:
                    return False, f"Memory service error: {str(e)}"
            
            self.monitoring_service.register_health_check("memory_service", check_memory_service)
    
    def index_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None
    ):
        """Index a document using the orchestrator and services"""
        return self.orchestrator.index_document(
            content=content,
            doc_id=doc_id,
            metadata=metadata,
            doc_type=doc_type
        )
    
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        evaluate: bool = False
    ) -> RAGResponse:
        """
        Main query method using orchestrator and services architecture
        """
        # Build configuration from instance config
        query_config = {
            'enable_query_optimization': self.config.enable_query_optimization,
            'enable_query_decomposition': self.config.enable_query_decomposition,
            'enable_hyde': self.config.enable_hyde,
            'retrieval_method': self.config.retrieval_method,
            'top_k': self.config.top_k,
            'enable_compression': self.config.enable_context_compression,
            'max_context_tokens': self.config.max_context_tokens,
            'use_reranking': self.config.use_reranking,
            'use_mmr': self.config.use_mmr,
            'mmr_lambda': self.config.mmr_lambda,
            'max_graph_hops': self.config.max_graph_hops,
            'experiment_id': self.config.experiment_id
        }
        
        return self.orchestrator.query(
            query=query,
            session_id=session_id,
            user_id=user_id,
            config=query_config,
            evaluate=evaluate
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status using orchestrator"""
        return self.orchestrator.get_system_status()
    
    def shutdown(self):
        """Graceful shutdown using orchestrator"""
        return self.orchestrator.shutdown()

# Backward compatibility alias for legacy code
# DEPRECATED: Use ProductionRAGSystem instead
class ProductionRAG(ProductionRAGSystem):
    """
    DEPRECATED: Legacy ProductionRAG class
    
    This class is provided for backward compatibility only.
    Please migrate to ProductionRAGSystem which provides the same API
    but with improved service-oriented architecture.
    
    Migration path:
    1. Replace ProductionRAG with ProductionRAGSystem
    2. Update configuration to use ProductionConfig instead of YAML files
    3. Use the new service-based architecture for better maintainability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize legacy ProductionRAG with backward compatibility
        
        Args:
            config_path: Path to legacy YAML config file (deprecated)
        """
        # Show deprecation warning
        import warnings
        warnings.warn(
            "ProductionRAG class is deprecated. Use ProductionRAGSystem instead. "
            "This legacy interface will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert legacy config if provided
        if config_path:
            legacy_config = self._load_legacy_config(config_path)
            modern_config = self._convert_legacy_config(legacy_config)
        else:
            modern_config = ProductionConfig()
        
        # Initialize with modern system
        super().__init__(modern_config)
    
    def _load_legacy_config(self, config_path: str) -> Dict[str, Any]:
        """Load legacy YAML configuration"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {}
    
    def _convert_legacy_config(self, legacy_config: Dict[str, Any]) -> ProductionConfig:
        """Convert legacy YAML config to modern ProductionConfig"""
        
        # Extract settings from legacy config structure
        embedding_config = legacy_config.get('embedding', {})
        retrieval_config = legacy_config.get('retrieval', {})
        generation_config = legacy_config.get('generation', {})
        
        return ProductionConfig(
            embedding_model=embedding_config.get('model', 'all-MiniLM-L6-v2'),
            reranker_model=retrieval_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
            llm_model=generation_config.get('model', 'gemma:2b'),
            retrieval_method=retrieval_config.get('method', 'adaptive'),
            top_k=retrieval_config.get('k_documents', 10),
            use_reranking=retrieval_config.get('rerank', True),
            enable_query_optimization=legacy_config.get('query_optimization', {}).get('enable', True),
            enable_knowledge_graph=legacy_config.get('knowledge_graph', {}).get('enable', True),
            enable_conversation_memory=legacy_config.get('conversation', {}).get('enable', True),
            enable_context_compression=legacy_config.get('context_compression', {}).get('enable', True),
            enable_monitoring=legacy_config.get('monitoring', {}).get('enable', True)
        )
    
    # Legacy API methods for backward compatibility
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        DEPRECATED: Legacy index_documents method
        
        Please use index_document() for individual documents instead.
        """
        warnings.warn(
            "index_documents() is deprecated. Use index_document() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        for i, doc_content in enumerate(documents):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            self.index_document(
                content=doc_content,
                doc_id=f"legacy_doc_{i}",
                metadata=doc_metadata
            )
    
    def query(self, request, use_cache: bool = True, evaluate: bool = False):
        """
        DEPRECATED: Legacy query method with different signature
        
        This method maintains backward compatibility with the old ProductionRAG API.
        """
        # Handle both legacy RAGRequest objects and simple strings
        if hasattr(request, 'query'):
            # RAGRequest object
            query_text = request.query
            session_id = getattr(request, 'session_id', None)
            user_id = getattr(request, 'user_id', None)
        else:
            # Simple string query
            query_text = str(request)
            session_id = None
            user_id = None
        
        # Call modern query method
        response = super().query(
            query=query_text,
            session_id=session_id,
            user_id=user_id,
            evaluate=evaluate
        )
        
        # Return in legacy RAGResponse format if needed
        return response