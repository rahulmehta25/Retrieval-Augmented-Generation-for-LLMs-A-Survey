import abc
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker(abc.ABC):
    """
    Abstract base class for reranking retrieved documents.
    """
    @abc.abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to the query.

        Args:
            query: The original user query.
            documents: A list of retrieved documents, each with at least a 'content' key.

        Returns:
            A list of documents, reranked by relevance score (highest first).
            Each document will have an additional 'relevance_score' key.
        """
        pass

class CrossEncoderReranker(Reranker):
    """
    Reranker using a Cross-Encoder model.
    Cross-encoders are typically more accurate than bi-encoders for relevance scoring
    because they process the query and document pair together.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            # Fallback to a simpler model if the specified one fails
            print(f"Warning: Could not load {model_name}, using fallback model")
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # Prepare sentence pairs for the cross-encoder
        sentence_pairs = [[query, doc["content"]] for doc in documents]

        try:
            # Get scores from the cross-encoder
            scores = self.model.predict(sentence_pairs)

            # Attach scores to documents and sort
            reranked_documents = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = float(scores[i])
                reranked_documents.append(doc_copy)

            # Sort in descending order of relevance score
            reranked_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
            return reranked_documents
            
        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Returning original order.")
            # Return documents with default scores if reranking fails
            for doc in documents:
                doc["relevance_score"] = 0.5
            return documents

class SimpleReranker(Reranker):
    """
    Simple reranker that uses basic heuristics for reranking.
    This is a fallback when cross-encoders are not available.
    """
    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        reranked_documents = []
        for doc in documents:
            doc_copy = doc.copy()
            content_lower = doc["content"].lower()
            content_words = set(content_lower.split())

            # Calculate simple relevance score based on word overlap
            overlap = len(query_words.intersection(content_words))
            total_query_words = len(query_words)
            
            if total_query_words > 0:
                relevance_score = overlap / total_query_words
            else:
                relevance_score = 0.0

            # Bonus for exact phrase matches
            if query_lower in content_lower:
                relevance_score += 0.5

            doc_copy["relevance_score"] = relevance_score
            reranked_documents.append(doc_copy)

        # Sort by relevance score
        reranked_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked_documents 