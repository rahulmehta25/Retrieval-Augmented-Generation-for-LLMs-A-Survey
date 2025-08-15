"""
RAG Module - Consolidated Production-Ready Implementation

This module provides a unified interface for all RAG implementations, with a focus on
the modern service-oriented architecture while maintaining backward compatibility.

RECOMMENDED USAGE:
==================

For new projects, use the modern service-oriented architecture:

from src.rag import ProductionRAGSystem, ProductionConfig

config = ProductionConfig(
    enable_query_optimization=True,
    enable_knowledge_graph=True,
    enable_conversation_memory=True,
    enable_context_compression=True
)
rag_system = ProductionRAGSystem(config)

# Index documents
rag_system.index_document("Your content here", "doc_id", {"type": "text"})

# Query with advanced features
response = rag_system.query("What is machine learning?")


LEGACY COMPATIBILITY:
=====================

For existing code, the deprecated ProductionRAG class is available:

from src.rag import ProductionRAG  # Shows deprecation warning

rag = ProductionRAG("config.yaml")  # Legacy YAML config support
response = rag.query(request)


OTHER RAG IMPLEMENTATIONS:
==========================

Educational and specialized implementations:

from src.rag import NaiveRAG, AdvancedRAG, ModularRAG

- NaiveRAG: Basic implementation for learning
- AdvancedRAG: Extends NaiveRAG with query optimization
- ModularRAG: Factory-pattern based modular architecture
"""

# Import the modern service-oriented implementation (recommended)
from .production_rag_integrated import (
    ProductionRAGSystem,
    ProductionConfig,
    RAGResponse
)

# Import backward compatibility class (deprecated but supported)
from .production_rag_integrated import ProductionRAG

# Import educational implementations
from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG

# Import factory for modular architecture
from .rag_factory import RAGComponentFactory

# Define what gets imported with "from src.rag import *"
__all__ = [
    # Modern recommended classes
    'ProductionRAGSystem',
    'ProductionConfig', 
    'RAGResponse',
    
    # Legacy compatibility (deprecated)
    'ProductionRAG',
    
    # Educational implementations
    'NaiveRAG',
    'AdvancedRAG', 
    'ModularRAG',
    
    # Factory
    'RAGComponentFactory'
]

# Module metadata
__version__ = "2.0.0"
__author__ = "RAG Development Team"
__description__ = "Production-ready RAG implementations with service-oriented architecture"

# Show migration message for common legacy imports
import warnings

def __getattr__(name):
    """Custom attribute access to show migration warnings for common legacy patterns"""
    
    if name in ['RAGRequest', 'RAGResponse']:
        warnings.warn(
            f"Importing {name} from production_rag is deprecated. "
            f"Use ProductionRAGSystem with the modern API instead. "
            f"See migration guide in __init__.py",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import from legacy module for compatibility
        if name == 'RAGRequest':
            from .production_rag import RAGRequest
            return RAGRequest
        elif name == 'RAGResponse':
            from .production_rag import RAGResponse as LegacyRAGResponse
            return LegacyRAGResponse
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")