"""
Hybrid Search Module

This module implements hybrid search strategies that combine dense vector search
with sparse keyword search for improved retrieval performance.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridSearch:
    """
    Hybrid search that combines dense vector search with sparse keyword search.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search.
        
        Args:
            alpha: Weight for dense search (1-alpha is weight for sparse search)
        """
        self.alpha = alpha
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
    
    def fit(self, documents: List[str]):
        """
        Fit the TF-IDF vectorizer on documents.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        if documents:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
    def search(self, query: str, dense_scores: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse scores.
        
        Args:
            query: Search query
            dense_scores: Scores from dense vector search
            k: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        if not self.documents or self.tfidf_matrix is None:
            # Fallback to dense search only
            return self._dense_only_search(dense_scores, k)
        
        # Get sparse scores
        sparse_scores = self._get_sparse_scores(query)
        
        # Normalize scores to [0, 1] range
        dense_scores_norm = self._normalize_scores(dense_scores)
        sparse_scores_norm = self._normalize_scores(sparse_scores)
        
        # Combine scores
        hybrid_scores = [
            self.alpha * dense + (1 - self.alpha) * sparse
            for dense, sparse in zip(dense_scores_norm, sparse_scores_norm)
        ]
        
        # Get top k results
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'content': self.documents[idx],
                'hybrid_score': hybrid_scores[idx],
                'dense_score': dense_scores_norm[idx],
                'sparse_score': sparse_scores_norm[idx]
            })
        
        return results
    
    def _get_sparse_scores(self, query: str) -> List[float]:
        """
        Get sparse search scores using TF-IDF.
        """
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            return similarities.tolist()
        except Exception as e:
            print(f"Warning: Sparse search failed: {e}")
            return [0.0] * len(self.documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        """
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _dense_only_search(self, dense_scores: List[float], k: int) -> List[Dict[str, Any]]:
        """
        Fallback to dense search only.
        """
        top_indices = np.argsort(dense_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'content': self.documents[idx] if idx < len(self.documents) else f"Document {idx}",
                'hybrid_score': dense_scores[idx],
                'dense_score': dense_scores[idx],
                'sparse_score': 0.0
            })
        
        return results

class KeywordBoostedSearch:
    """
    Search that boosts documents containing query keywords.
    """
    
    def __init__(self, keyword_boost: float = 2.0):
        """
        Initialize keyword-boosted search.
        
        Args:
            keyword_boost: Multiplier for documents containing keywords
        """
        self.keyword_boost = keyword_boost
    
    def search(self, query: str, documents: List[str], dense_scores: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform search with keyword boosting.
        
        Args:
            query: Search query
            documents: List of document texts
            dense_scores: Scores from dense vector search
            k: Number of results to return
            
        Returns:
            List of search results with boosted scores
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Calculate keyword scores
        keyword_scores = []
        for doc in documents:
            score = self._calculate_keyword_score(doc, keywords)
            keyword_scores.append(score)
        
        # Normalize scores
        dense_scores_norm = self._normalize_scores(dense_scores)
        keyword_scores_norm = self._normalize_scores(keyword_scores)
        
        # Combine scores with keyword boosting
        boosted_scores = [
            dense * (1 + (self.keyword_boost - 1) * keyword)
            for dense, keyword in zip(dense_scores_norm, keyword_scores_norm)
        ]
        
        # Get top k results
        top_indices = np.argsort(boosted_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'content': documents[idx],
                'boosted_score': boosted_scores[idx],
                'dense_score': dense_scores_norm[idx],
                'keyword_score': keyword_scores_norm[idx]
            })
        
        return results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_keyword_score(self, document: str, keywords: List[str]) -> float:
        """
        Calculate keyword match score for a document.
        """
        if not keywords:
            return 0.0
        
        doc_lower = document.lower()
        matches = sum(1 for keyword in keywords if keyword in doc_lower)
        
        # Normalize by number of keywords
        return matches / len(keywords)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        """
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

class MultiVectorSearch:
    """
    Search using multiple embedding models for different aspects.
    """
    
    def __init__(self, embedders: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize multi-vector search.
        
        Args:
            embedders: Dictionary of embedder instances
            weights: Weights for each embedder (default: equal weights)
        """
        self.embedders = embedders
        self.weights = weights or {name: 1.0/len(embedders) for name in embedders.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
    
    def search(self, query: str, document_embeddings: Dict[str, List[List[float]]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform multi-vector search.
        
        Args:
            query: Search query
            document_embeddings: Dictionary mapping embedder names to document embeddings
            k: Number of results to return
            
        Returns:
            List of search results with combined scores
        """
        # Get query embeddings for each embedder
        query_embeddings = {}
        for name, embedder in self.embedders.items():
            try:
                query_emb = embedder.embed([query])[0]
                query_embeddings[name] = query_emb
            except Exception as e:
                print(f"Warning: Failed to get embedding from {name}: {e}")
                query_embeddings[name] = None
        
        # Calculate scores for each embedder
        all_scores = []
        for name, query_emb in query_embeddings.items():
            if query_emb is None or name not in document_embeddings:
                continue
                
            doc_embs = document_embeddings[name]
            scores = self._calculate_similarities(query_emb, doc_embs)
            all_scores.append((name, scores))
        
        if not all_scores:
            return []
        
        # Combine scores using weights
        combined_scores = [0.0] * len(all_scores[0][1])
        for name, scores in all_scores:
            weight = self.weights.get(name, 0.0)
            for i, score in enumerate(scores):
                combined_scores[i] += weight * score
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            result = {
                'index': int(idx),
                'combined_score': combined_scores[idx]
            }
            
            # Add individual scores
            for name, scores in all_scores:
                result[f'{name}_score'] = scores[idx]
            
            results.append(result)
        
        return results
    
    def _calculate_similarities(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarities between query and documents.
        """
        similarities = []
        for doc_emb in document_embeddings:
            similarity = self._cosine_similarity(query_embedding, doc_emb)
            similarities.append(similarity)
        return similarities
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2) 