"""
Advanced Hybrid Retrieval with BM25 + Dense Vectors + Reranking
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval result"""
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 'dense', 'sparse', or 'hybrid'
    doc_id: str
    chunk_id: str

class AdvancedHybridRetriever:
    """
    Production hybrid retriever with:
    - BM25 for sparse retrieval
    - Dense embeddings with multiple indices
    - Cross-encoder reranking
    - Query-adaptive fusion
    - Diversity-aware retrieval
    - Caching and optimization
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_path: Optional[str] = "./retrieval_index"
    ):
        """Initialize hybrid retriever"""
        
        # Models
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        
        # Storage paths
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # BM25 components
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []
        
        # Dense index (FAISS)
        self.faiss_index = None
        self.faiss_metadata = []
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # ChromaDB for metadata-rich retrieval
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.index_path / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.chroma_collection = None
        
        # TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.tfidf_matrix = None
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "index_size_mb": 0
        }
        
        # Cache
        self.query_cache = {}
        self.embedding_cache = {}
        
        # Load existing indices
        self.load_indices()
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """Add documents to all indices"""
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Prepare for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Update BM25
        if self.bm25_index is None:
            self.bm25_corpus = tokenized_docs
            self.bm25_index = BM25Okapi(tokenized_docs)
        else:
            self.bm25_corpus.extend(tokenized_docs)
            self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Update BM25 metadata
        for i, (doc, doc_id, meta) in enumerate(zip(documents, doc_ids, metadata)):
            self.bm25_metadata.append({
                "text": doc,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                **meta
            })
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            documents,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Update FAISS index
        if self.faiss_index is None:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Also create IVF index for large-scale retrieval
            if len(documents) > 10000:
                nlist = int(np.sqrt(len(documents)))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, nlist
                )
                self.faiss_index.train(embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        # Update FAISS metadata
        for i, (doc, doc_id, meta) in enumerate(zip(documents, doc_ids, metadata)):
            self.faiss_metadata.append({
                "text": doc,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                **meta
            })
        
        # Update ChromaDB
        if self.chroma_collection is None:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="hybrid_retrieval",
                metadata={"description": "Hybrid retrieval collection"}
            )
        
        self.chroma_collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=[f"{doc_id}_chunk_{i}" for i, doc_id in enumerate(doc_ids)],
            metadatas=metadata
        )
        
        # Update TF-IDF
        if self.tfidf_matrix is None:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        else:
            # Refit on all documents
            all_docs = [meta["text"] for meta in self.bm25_metadata]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs)
        
        # Update statistics
        self.stats["total_documents"] = len(set(doc_ids))
        self.stats["total_chunks"] = len(self.bm25_metadata)
        self.stats["avg_chunk_length"] = np.mean([len(doc.split()) for doc in documents])
        
        logger.info(f"Added {len(documents)} documents to hybrid indices")
        
        # Save indices
        self.save_indices()
    
    def sparse_retrieval(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """BM25 sparse retrieval"""
        
        if self.bm25_index is None:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                meta = self.bm25_metadata[idx]
                results.append(RetrievalResult(
                    text=meta["text"],
                    score=float(scores[idx]),
                    metadata=meta,
                    source="sparse",
                    doc_id=meta["doc_id"],
                    chunk_id=meta["chunk_id"]
                ))
        
        return results
    
    def dense_retrieval(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Dense embedding retrieval"""
        
        if self.faiss_index is None:
            return []
        
        # Check cache
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            # Encode query
            query_embedding = self.embedder.encode(query)
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            self.embedding_cache[query] = query_embedding
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.faiss_metadata):
                meta = self.faiss_metadata[idx]
                results.append(RetrievalResult(
                    text=meta["text"],
                    score=float(dist),
                    metadata=meta,
                    source="dense",
                    doc_id=meta["doc_id"],
                    chunk_id=meta["chunk_id"]
                ))
        
        return results
    
    def keyword_retrieval(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """TF-IDF keyword-based retrieval"""
        
        if self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                meta = self.bm25_metadata[idx]
                results.append(RetrievalResult(
                    text=meta["text"],
                    score=float(similarities[idx]),
                    metadata=meta,
                    source="keyword",
                    doc_id=meta["doc_id"],
                    chunk_id=meta["chunk_id"]
                ))
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        k: int = 60,
        weights: Optional[List[float]] = None
    ) -> List[RetrievalResult]:
        """Fuse multiple result lists using RRF"""
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        for weight, results in zip(weights, result_lists):
            for rank, result in enumerate(results):
                rrf_scores[result.chunk_id] += weight / (k + rank + 1)
                doc_map[result.chunk_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id, score in sorted_ids:
            result = doc_map[chunk_id]
            result.score = score
            result.source = "hybrid"
            fused_results.append(result)
        
        return fused_results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        
        if not results:
            return []
        
        # Prepare pairs
        pairs = [[query, result.text] for result in results]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs, batch_size=16)
        
        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def maximal_marginal_relevance(
        self,
        query: str,
        results: List[RetrievalResult],
        lambda_param: float = 0.5,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Apply MMR for diversity"""
        
        if not results:
            return []
        
        # Encode query and documents
        query_embedding = self.embedder.encode(query)
        doc_embeddings = self.embedder.encode([r.text for r in results])
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        query_similarities = np.dot(doc_embeddings, query_embedding)
        doc_similarities = np.dot(doc_embeddings, doc_embeddings.T)
        
        # MMR selection
        selected = []
        remaining = list(range(len(results)))
        
        while len(selected) < top_k and remaining:
            if not selected:
                # Select most similar to query
                idx = np.argmax(query_similarities[remaining])
                best_idx = remaining[idx]
            else:
                # Calculate MMR scores
                mmr_scores = []
                for idx in remaining:
                    relevance = query_similarities[idx]
                    max_sim = np.max([doc_similarities[idx][s] for s in selected])
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr)
                
                best_idx = remaining[np.argmax(mmr_scores)]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [results[i] for i in selected]
    
    def hybrid_retrieve(
        self,
        query: str,
        k: int = 10,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.5,
        keyword_weight: float = 0.2,
        use_reranking: bool = True,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5
    ) -> List[RetrievalResult]:
        """Main hybrid retrieval method"""
        
        # Check cache
        cache_key = f"{query}_{k}_{sparse_weight}_{dense_weight}_{use_reranking}_{use_mmr}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Get results from each method
        sparse_results = self.sparse_retrieval(query, k * 2)
        dense_results = self.dense_retrieval(query, k * 2)
        keyword_results = self.keyword_retrieval(query, k * 2)
        
        # Fuse results
        all_results = [sparse_results, dense_results, keyword_results]
        weights = [sparse_weight, dense_weight, keyword_weight]
        
        fused_results = self.reciprocal_rank_fusion(all_results, weights=weights)
        
        # Rerank if requested
        if use_reranking and fused_results:
            fused_results = self.rerank(query, fused_results[:k*2], top_k=k*2)
        
        # Apply MMR if requested
        if use_mmr:
            fused_results = self.maximal_marginal_relevance(
                query, fused_results, lambda_param=mmr_lambda, top_k=k
            )
        else:
            fused_results = fused_results[:k]
        
        # Cache results
        self.query_cache[cache_key] = fused_results
        
        return fused_results
    
    def adaptive_retrieve(
        self,
        query: str,
        k: int = 10
    ) -> List[RetrievalResult]:
        """Adaptive retrieval that adjusts weights based on query type"""
        
        # Analyze query
        query_length = len(query.split())
        has_keywords = any(word in query.lower() for word in ["specific", "exact", "precisely"])
        is_question = query.strip().endswith("?")
        
        # Adjust weights
        if query_length < 5:
            # Short queries benefit from keyword matching
            sparse_weight = 0.5
            dense_weight = 0.3
            keyword_weight = 0.2
        elif has_keywords:
            # Specific queries need exact matching
            sparse_weight = 0.6
            dense_weight = 0.2
            keyword_weight = 0.2
        elif is_question:
            # Questions benefit from semantic understanding
            sparse_weight = 0.2
            dense_weight = 0.6
            keyword_weight = 0.2
        else:
            # Default balanced
            sparse_weight = 0.33
            dense_weight = 0.34
            keyword_weight = 0.33
        
        return self.hybrid_retrieve(
            query, k,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight
        )
    
    def save_indices(self):
        """Save all indices to disk"""
        
        # Save BM25
        bm25_path = self.index_path / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                "corpus": self.bm25_corpus,
                "metadata": self.bm25_metadata
            }, f)
        
        # Save FAISS
        if self.faiss_index:
            faiss_path = self.index_path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            # Save metadata
            meta_path = self.index_path / "faiss_metadata.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)
        
        # Save TF-IDF
        if self.tfidf_matrix is not None:
            tfidf_path = self.index_path / "tfidf.pkl"
            with open(tfidf_path, 'wb') as f:
                pickle.dump({
                    "vectorizer": self.tfidf_vectorizer,
                    "matrix": self.tfidf_matrix
                }, f)
        
        logger.info("Indices saved")
    
    def load_indices(self):
        """Load indices from disk"""
        
        # Load BM25
        bm25_path = self.index_path / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_corpus = data["corpus"]
                self.bm25_metadata = data["metadata"]
                if self.bm25_corpus:
                    self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Load FAISS
        faiss_path = self.index_path / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            meta_path = self.index_path / "faiss_metadata.pkl"
            with open(meta_path, 'rb') as f:
                self.faiss_metadata = pickle.load(f)
        
        # Load TF-IDF
        tfidf_path = self.index_path / "tfidf.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                data = pickle.load(f)
                self.tfidf_vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
        
        # Load ChromaDB
        try:
            self.chroma_collection = self.chroma_client.get_collection("hybrid_retrieval")
        except:
            pass
        
        if self.bm25_metadata:
            logger.info(f"Loaded indices with {len(self.bm25_metadata)} documents")