"""
Retrieval Service - Handles document retrieval and context processing
"""

import logging
import time
from typing import List, Dict, Any, Optional
from .interfaces import RetrievalServiceInterface, RetrievalResult, MemoryContext
from ..retrieval.advanced_hybrid_retriever import AdvancedHybridRetriever
from ..retrieval.advanced_context_compressor import AdvancedContextCompressor
from ..graph_rag.advanced_knowledge_graph import AdvancedKnowledgeGraph, GraphQuery
from ..chunking.semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service responsible for document retrieval and context processing
    Implements single responsibility principle for retrieval operations
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_path: str = "./rag_index",
        knowledge_graph_path: Optional[str] = None,
        max_context_tokens: int = 2000
    ):
        """Initialize retrieval service components"""
        self.hybrid_retriever = AdvancedHybridRetriever(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            index_path=index_path
        )
        
        self.context_compressor = AdvancedContextCompressor(
            embedding_model=embedding_model,
            max_tokens=max_context_tokens
        )
        
        self.semantic_chunker = SemanticChunker()
        
        # Optional knowledge graph
        self.knowledge_graph = None
        if knowledge_graph_path:
            self.knowledge_graph = AdvancedKnowledgeGraph(
                persist_path=knowledge_graph_path
            )
        
        logger.info("RetrievalService initialized successfully")
    
    def retrieve_contexts(
        self,
        query: str,
        sub_queries: List[str],
        entities: List[str],
        conversation_contexts: List[MemoryContext],
        retrieval_method: str = "adaptive",
        top_k: int = 10,
        enable_compression: bool = True,
        max_context_tokens: int = 2000,
        use_reranking: bool = True,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        max_graph_hops: int = 2
    ) -> RetrievalResult:
        """
        Retrieve and process contexts from multiple sources
        
        Args:
            query: Main query
            sub_queries: List of decomposed queries
            entities: Extracted entities for graph search
            conversation_contexts: Context from conversation memory
            retrieval_method: Method to use for retrieval
            top_k: Number of contexts to retrieve
            enable_compression: Whether to compress contexts
            max_context_tokens: Maximum tokens in final context
            use_reranking: Whether to use reranking
            use_mmr: Whether to use MMR diversity
            mmr_lambda: MMR lambda parameter
            max_graph_hops: Maximum hops for graph exploration
            
        Returns:
            RetrievalResult with processed contexts
        """
        start_time = time.time()
        logger.info(f"Retrieving contexts for query: {query[:50]}...")
        
        try:
            # 1. Hybrid Retrieval from vector store
            all_contexts = []
            retrieval_scores = []
            
            for sub_query in sub_queries:
                if retrieval_method == "adaptive":
                    results = self.hybrid_retriever.adaptive_retrieve(sub_query, k=top_k)
                elif retrieval_method == "hybrid":
                    results = self.hybrid_retriever.hybrid_retrieve(
                        sub_query,
                        k=top_k,
                        use_reranking=use_reranking,
                        use_mmr=use_mmr,
                        mmr_lambda=mmr_lambda
                    )
                elif retrieval_method == "sparse":
                    results = self.hybrid_retriever.sparse_retrieval(sub_query, k=top_k)
                else:  # dense
                    results = self.hybrid_retriever.dense_retrieval(sub_query, k=top_k)
                
                all_contexts.extend([r.text for r in results])
                retrieval_scores.extend([r.score for r in results])
            
            # 2. Knowledge Graph Exploration
            graph_contexts = []
            graph_entities_found = 0
            graph_relations_used = 0
            
            if self.knowledge_graph and entities:
                graph_query = GraphQuery(
                    entities=entities[:3],
                    relations=[],
                    max_hops=max_graph_hops,
                    limit=5
                )
                
                graph_results = self.knowledge_graph.query_graph(graph_query)
                graph_entities_found = len(graph_results['entities'])
                graph_relations_used = len(graph_results['relations'])
                
                # Convert to contexts
                for entity in graph_results['entities'][:3]:
                    graph_contexts.append(f"{entity['type']}: {entity['text']}")
            
            # 3. Context Compression
            compression_ratio = None
            contexts_compressed = False
            
            if enable_compression and all_contexts:
                # Combine all context sources
                combined_contexts = (
                    all_contexts +
                    [c.content for c in conversation_contexts] +
                    graph_contexts
                )
                
                if len(combined_contexts) > 0:
                    compressed = self.context_compressor.adaptive_compression(
                        combined_contexts,
                        query,
                        max_context_tokens
                    )
                    
                    final_contexts = [compressed.compressed_text]
                    compression_ratio = compressed.compression_ratio
                    contexts_compressed = True
                else:
                    final_contexts = all_contexts[:5]
            else:
                final_contexts = all_contexts[:5]
            
            # Prepare metadata
            metadata = []
            for i, context in enumerate(final_contexts):
                meta = {
                    "index": i,
                    "source": "vector_store",
                    "compressed": contexts_compressed
                }
                if i < len(retrieval_scores):
                    meta["score"] = retrieval_scores[i]
                metadata.append(meta)
            
            result = RetrievalResult(
                contexts=final_contexts,
                scores=retrieval_scores[:len(final_contexts)],
                metadata=metadata,
                graph_entities_found=graph_entities_found,
                graph_relations_used=graph_relations_used,
                compression_ratio=compression_ratio,
                contexts_compressed=contexts_compressed
            )
            
            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Context retrieval completed in {retrieval_time:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving contexts: {e}")
            # Return empty result on error
            return RetrievalResult(
                contexts=[],
                scores=[],
                metadata=[],
                graph_entities_found=0,
                graph_relations_used=0,
                compression_ratio=None,
                contexts_compressed=False
            )
    
    def index_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None
    ) -> None:
        """
        Index a document with semantic chunking
        
        Args:
            content: Document content
            doc_id: Unique document identifier
            metadata: Optional metadata
            doc_type: Type of document for chunking optimization
        """
        logger.info(f"Indexing document: {doc_id}")
        
        try:
            # Chunk document with semantic boundaries
            chunks = self.semantic_chunker.smart_chunk(content, doc_type)
            logger.info(f"Document {doc_id} chunked into {len(chunks)} segments")
            
            # Add to hybrid retriever
            self.hybrid_retriever.add_documents(
                documents=[chunk.text for chunk in chunks],
                doc_ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
                metadata=[{**(metadata or {}), "chunk_id": i} for i in range(len(chunks))]
            )
            
            # Add to knowledge graph if available
            if self.knowledge_graph:
                for i, chunk in enumerate(chunks[:20]):  # Limit for performance
                    self.knowledge_graph.add_document(
                        chunk.text,
                        f"{doc_id}_chunk_{i}",
                        metadata=metadata
                    )
            
            logger.info(f"Document {doc_id} indexed successfully")
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            raise
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_retrieval_time": 0.0
        }
        
        # Add hybrid retriever stats if available
        if hasattr(self.hybrid_retriever, 'stats'):
            stats.update(self.hybrid_retriever.stats)
        
        # Add knowledge graph stats if available
        if self.knowledge_graph:
            graph_stats = self.knowledge_graph.get_statistics()
            stats.update({
                "graph_entities": graph_stats.get('total_entities', 0),
                "graph_relations": graph_stats.get('total_relations', 0)
            })
        
        return stats