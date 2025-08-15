"""
Query Service - Handles query optimization and processing
"""

import logging
from typing import List, Optional
from .interfaces import QueryServiceInterface, QueryOptimizationResult
from ..optimization.semantic_query_optimizer import SemanticQueryOptimizer, QueryRewriter

logger = logging.getLogger(__name__)

class QueryService:
    """
    Service responsible for query optimization and processing
    Implements single responsibility principle for query-related operations
    """
    
    def __init__(self):
        """Initialize query service components"""
        self.query_optimizer = SemanticQueryOptimizer()
        self.query_rewriter = QueryRewriter()
        logger.info("QueryService initialized successfully")
    
    def optimize_query(
        self,
        query: str,
        enable_decomposition: bool = True,
        enable_hyde: bool = True
    ) -> QueryOptimizationResult:
        """
        Optimize query with various techniques
        
        Args:
            query: Original user query
            enable_decomposition: Whether to decompose complex queries
            enable_hyde: Whether to generate hypothetical documents
            
        Returns:
            QueryOptimizationResult with optimized query variants
        """
        logger.info(f"Optimizing query: {query[:50]}...")
        
        try:
            # Semantic optimization
            semantic_query = self.query_optimizer.optimize(query)
            
            optimized_query = semantic_query.cleaned
            query_intent = semantic_query.intent
            query_complexity = semantic_query.complexity
            query_entities = semantic_query.keywords
            
            # Query decomposition for complex queries
            sub_queries = [optimized_query]  # Default to single query
            if enable_decomposition and query_complexity and query_complexity > 0.7:
                sub_queries = semantic_query.sub_queries
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            
            # HyDE generation for specific intents
            hyde_query = None
            if enable_hyde and query_intent in ["definition", "explanation"]:
                hyde_query = self.query_rewriter.generate_hypothetical_document(optimized_query)
                logger.info("Generated HyDE query")
            
            result = QueryOptimizationResult(
                original_query=query,
                optimized_query=optimized_query,
                hyde_query=hyde_query,
                sub_queries=sub_queries,
                intent=query_intent,
                complexity=query_complexity,
                entities=query_entities
            )
            
            logger.info("Query optimization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            # Return fallback result
            return QueryOptimizationResult(
                original_query=query,
                optimized_query=query,
                hyde_query=None,
                sub_queries=[query],
                intent=None,
                complexity=None,
                entities=[]
            )
    
    def get_query_statistics(self) -> dict:
        """Get query processing statistics"""
        # This could be extended with actual statistics tracking
        return {
            "total_queries_processed": 0,  # Would track actual counts
            "avg_complexity": 0.0,
            "common_intents": []
        }