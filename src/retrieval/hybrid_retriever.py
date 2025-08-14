"""
Production-grade Hybrid Retrieval combining dense and sparse methods
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.utils import embedding_functions
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata"""
    text: str
    score: float
    doc_id: str
    chunk_id: int
    metadata: Dict[str, Any]
    
class HybridRetriever:
    """
    Production hybrid retriever combining:
    1. Dense retrieval (semantic search)
    2. Sparse retrieval (BM25 keyword search)
    3. Cross-encoder reranking
    4. Reciprocal Rank Fusion
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "hybrid_rag",
        persist_dir: str = "./chroma_hybrid_db"
    ):
        """Initialize hybrid retriever with dense and sparse components"""
        
        # Dense retrieval setup
        self.embedder = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model
                )
            )
        
        # Reranker setup
        self.reranker = CrossEncoder(reranker_model)
        
        # Sparse retrieval setup
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
        logger.info(f"Initialized HybridRetriever with {embedding_model}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """Add documents to both dense and sparse indices"""
        
        if not documents:
            return
        
        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        # Add to dense index (ChromaDB)
        self.collection.add(
            documents=documents,
            ids=doc_ids,
            metadatas=metadata
        )
        
        # Add to sparse index (BM25)
        self.documents.extend(documents)
        self.doc_metadata.extend(list(zip(doc_ids, metadata)))
        
        # Rebuild BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Added {len(documents)} documents to hybrid index")
    
    def dense_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Perform dense semantic search"""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        retrieval_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieval_results.append(RetrievalResult(
                    text=doc,
                    score=1.0 - results['distances'][0][i] if results['distances'] else 1.0,
                    doc_id=results['ids'][0][i],
                    chunk_id=i,
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                ))
        
        return retrieval_results
    
    def sparse_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Perform sparse BM25 search"""
        
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        retrieval_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id, metadata = self.doc_metadata[idx] if idx < len(self.doc_metadata) else ("unknown", {})
                retrieval_results.append(RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    doc_id=doc_id,
                    chunk_id=idx,
                    metadata=metadata
                ))
        
        return retrieval_results
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int = 60,
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion
        RRF score = sum(1 / (k + rank))
        """
        
        # Create score dictionaries
        rrf_scores = {}
        doc_map = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_key = result.text[:100]  # Use first 100 chars as key
            rrf_scores[doc_key] = alpha * (1 / (k + rank + 1))
            doc_map[doc_key] = result
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_key = result.text[:100]
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += (1 - alpha) * (1 / (k + rank + 1))
            else:
                rrf_scores[doc_key] = (1 - alpha) * (1 / (k + rank + 1))
                doc_map[doc_key] = result
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return combined results
        combined_results = []
        for doc_key, score in sorted_docs:
            result = doc_map[doc_key]
            result.score = score
            combined_results.append(result)
        
        return combined_results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        k: int = 5
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, result.text] for result in results]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Update scores and sort
        for i, result in enumerate(results):
            result.score = float(scores[i])
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_reranking: bool = True,
        alpha: float = 0.5
    ) -> List[str]:
        """
        Main retrieval method combining all techniques
        
        Args:
            query: Search query
            k: Number of results to return
            use_reranking: Whether to apply cross-encoder reranking
            alpha: Weight for dense search (0=sparse only, 1=dense only)
        
        Returns:
            List of retrieved text chunks
        """
        
        # Step 1: Dense search
        dense_results = self.dense_search(query, k=k*3)
        
        # Step 2: Sparse search
        sparse_results = self.sparse_search(query, k=k*3)
        
        # Step 3: Reciprocal Rank Fusion
        combined_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=alpha
        )
        
        # Step 4: Reranking (optional)
        if use_reranking and combined_results:
            final_results = self.rerank(query, combined_results[:k*2], k=k)
        else:
            final_results = combined_results[:k]
        
        # Return text only
        return [r.text for r in final_results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_documents": len(self.documents),
            "collection_count": self.collection.count(),
            "has_bm25_index": self.bm25 is not None,
            "embedding_model": self.embedder.get_sentence_embedding_dimension(),
            "reranker_available": self.reranker is not None
        }

class SmartChunker:
    """
    Intelligent chunking with semantic boundaries
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_splitter = None
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            self.sentence_splitter = sent_tokenize
        except:
            logger.warning("NLTK not available, using simple splitting")
    
    def chunk_with_sentences(self, text: str) -> List[str]:
        """Chunk text respecting sentence boundaries"""
        
        if not self.sentence_splitter:
            # Fallback to simple splitting
            return self.simple_chunk(text)
        
        sentences = self.sentence_splitter(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def simple_chunk(self, text: str) -> List[str]:
        """Simple overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        
        return chunks
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk document and add metadata"""
        
        chunks = self.chunk_with_sentences(text)
        
        results = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": i,
                "chunk_count": len(chunks),
                **(metadata or {})
            }
            results.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return results