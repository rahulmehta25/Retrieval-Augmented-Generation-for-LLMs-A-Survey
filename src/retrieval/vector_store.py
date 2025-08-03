import abc
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

class VectorStore(abc.ABC):
    """
    Abstract base class for vector store implementations.
    """
    @abc.abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Adds documents and their embeddings to the vector store.
        """
        pass

    @abc.abstractmethod
    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search.

        Args:
            query_embedding: The embedding of the query.
            k: The number of top similar documents to retrieve.
            filters: Optional metadata filters.

        Returns:
            A list of dictionaries, each representing a retrieved document with content and metadata.
        """
        pass

    @abc.abstractmethod
    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        """
        Updates an existing document in the vector store.
        """
        pass

    @abc.abstractmethod
    def delete_document(self, doc_id: str):
        """
        Deletes a document from the vector store.
        """
        pass

class FAISSVectorStore(VectorStore):
    """
    Vector store implementation using FAISS.
    FAISS is an open-source library for efficient similarity search and clustering of dense vectors.
    """
    def __init__(self, embedding_dimension: int):
        self.index = faiss.IndexFlatL2(embedding_dimension) # L2 distance for similarity
        self.documents = [] # Stores {'id': ..., 'content': ..., 'metadata': ..., 'embedding': ...}
        self.doc_id_map = {}
        self.next_id = 0

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not documents or not embeddings:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        new_embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(new_embeddings_np)

        for i, doc in enumerate(documents):
            doc_id = str(self.next_id)
            self.documents.append({
                'id': doc_id,
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'embedding': embeddings[i]
            })
            self.doc_id_map[doc_id] = self.next_id # Map doc_id to internal FAISS index
            self.next_id += 1

    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_np, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: # FAISS returns -1 for empty slots if k > num_vectors
                continue
            doc = self.documents[idx]
            # Apply filters if provided (FAISS itself doesn't support direct metadata filtering)
            if filters:
                match = True
                for key, value in filters.items():
                    if doc['metadata'].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            results.append({"content": doc['content'], "metadata": doc['metadata'], "distance": distances[0][i]})
        return results

    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        # FAISS does not support direct update/delete of individual vectors easily.
        # A common strategy is to rebuild the index or mark documents as 'deleted'.
        # For simplicity, this implementation will not fully support in-place updates/deletes
        # without rebuilding the index, which is inefficient for single updates.
        # For a production system, consider a vector database that supports these operations natively.
        print(f"Warning: FAISSVectorStore does not efficiently support updating/deleting individual documents. "
              f"Consider rebuilding the index or using a different vector store for frequent updates.")

    def delete_document(self, doc_id: str):
        print(f"Warning: FAISSVectorStore does not efficiently support updating/deleting individual documents. "
              f"Consider rebuilding the index or using a different vector store for frequent deletions.")

class ChromaDBVectorStore(VectorStore):
    """
    Vector store implementation using ChromaDB.
    ChromaDB is an open-source embedding database that makes it easy to build LLM apps.
    """
    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.client = chromadb.PersistentClient(path=path)
        # ChromaDB requires an embedding function, which it uses internally.
        # We'll use a dummy one here since embeddings are provided externally.
        # In a real scenario, you might let ChromaDB handle embeddings or use a consistent one.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not documents or not embeddings:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
            
        ids = [str(i) for i in range(self.collection.count(), self.collection.count() + len(documents))]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_kwargs = {
            'query_embeddings': [query_embedding],
            'n_results': k,
            'include': ['documents', 'metadatas', 'distances']
        }
        if filters:
            query_kwargs['where'] = filters
            
        results = self.collection.query(**query_kwargs)
        retrieved_docs = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return retrieved_docs

    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        update_data = {}
        if new_content is not None: update_data['documents'] = [new_content]
        if new_metadata is not None: update_data['metadatas'] = [new_metadata]
        if new_embedding is not None: update_data['embeddings'] = [new_embedding]
        self.collection.update(ids=[doc_id], **update_data)

    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id]) 