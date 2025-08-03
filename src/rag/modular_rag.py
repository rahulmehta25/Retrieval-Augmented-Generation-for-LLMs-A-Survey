"""
Modular RAG Architecture

This module implements a modular RAG system that uses the factory pattern
and supports advanced features like context compression, hybrid search,
and conversation memory.
"""

import os
import yaml
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.rag.rag_factory import RAGComponentFactory
from src.retrieval.context_compressor import (
    ExtractiveContextCompressor, 
    AbstractiveContextCompressor,
    HybridContextCompressor
)
from src.retrieval.hybrid_search import HybridSearch, KeywordBoostedSearch
from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationMemory:
    """
    Manages conversation history and context for multi-turn dialogues.
    """
    
    def __init__(self, max_turns: int = 10, max_tokens: int = 2000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.conversation_history = []
    
    def add_turn(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]] = None):
        """
        Add a conversation turn to memory.
        """
        turn = {
            'timestamp': datetime.now(),
            'query': query,
            'answer': answer,
            'retrieved_docs': retrieved_docs or []
        }
        
        self.conversation_history.append(turn)
        
        # Maintain max turns
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)
    
    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.
        """
        return self.conversation_history[-num_turns:] if self.conversation_history else []
    
    def get_summary_context(self) -> str:
        """
        Get a summary of the conversation for context.
        """
        if not self.conversation_history:
            return ""
        
        summary_parts = []
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            summary_parts.append(f"Q: {turn['query']}")
            summary_parts.append(f"A: {turn['answer'][:100]}...")
        
        return "\n".join(summary_parts)
    
    def clear(self):
        """
        Clear conversation history.
        """
        self.conversation_history.clear()

class ModularRAG:
    """
    Modular RAG system with factory pattern and advanced features.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize modular RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.components = self._initialize_components()
        self.conversation_memory = ConversationMemory()
        
        # Initialize advanced features
        self.context_compressor = self._initialize_context_compressor()
        self.hybrid_search = self._initialize_hybrid_search()
        self.reranker = self._initialize_reranker()
        
        logging.info("Modular RAG system initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """
        Get default configuration.
        """
        return {
            'text_splitter': {'type': 'fixed_size', 'chunk_size': 500, 'chunk_overlap': 50},
            'embedder': {'type': 'sentence_transformer', 'model_name': 'all-MiniLM-L6-v2'},
            'vector_store': {'type': 'chromadb', 'path': './chroma_db'},
            'generator': {'type': 'huggingface', 'model_name': 'distilgpt2'},
            'reranker': {'type': 'cross_encoder'},
            'context_compression': {'type': 'extractive'},
            'hybrid_search': {'type': 'hybrid', 'alpha': 0.5}
        }
    
    def _initialize_components(self) -> Dict[str, Any]:
        """
        Initialize RAG components using factory pattern.
        """
        try:
            components = RAGComponentFactory.create_rag_system(self.config)
            logging.info("RAG components initialized successfully")
            return components
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_context_compressor(self) -> Any:
        """
        Initialize context compressor based on configuration.
        """
        compression_config = self.config.get('context_compression', {})
        compression_type = compression_config.get('type', 'extractive')
        
        if compression_type == 'extractive':
            return ExtractiveContextCompressor()
        elif compression_type == 'abstractive':
            return AbstractiveContextCompressor()
        elif compression_type == 'hybrid':
            return HybridContextCompressor()
        else:
            logging.warning(f"Unknown compression type: {compression_type}, using extractive")
            return ExtractiveContextCompressor()
    
    def _initialize_hybrid_search(self) -> Any:
        """
        Initialize hybrid search based on configuration.
        """
        search_config = self.config.get('hybrid_search', {})
        search_type = search_config.get('type', 'hybrid')
        
        if search_type == 'hybrid':
            alpha = search_config.get('alpha', 0.5)
            return HybridSearch(alpha=alpha)
        elif search_type == 'keyword_boost':
            boost = search_config.get('keyword_boost', 2.0)
            return KeywordBoostedSearch(keyword_boost=boost)
        else:
            logging.warning(f"Unknown search type: {search_type}, using hybrid")
            return HybridSearch()
    
    def _initialize_reranker(self) -> Any:
        """
        Initialize reranker based on configuration.
        """
        reranker_config = self.config.get('reranker', {})
        reranker_type = reranker_config.get('type', 'cross_encoder')
        
        if reranker_type == 'cross_encoder':
            return CrossEncoderReranker()
        elif reranker_type == 'simple':
            return SimpleReranker()
        else:
            logging.warning(f"Unknown reranker type: {reranker_type}, using simple")
            return SimpleReranker()
    
    def index_documents(self, file_paths: List[str]) -> None:
        """
        Index documents using modular components.
        """
        logging.info(f"Indexing {len(file_paths)} documents...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                # Load and chunk document
                chunks = self.components['text_splitter'].split_text_from_file(file_path)
                all_chunks.extend(chunks)
                logging.info(f"Processed {len(chunks)} chunks from {file_path}")
            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {e}")
        
        if not all_chunks:
            logging.warning("No chunks to index")
            return
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        logging.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = self.components['embedder'].embed(chunk_texts)
        
        # Add to vector store
        logging.info(f"Adding {len(all_chunks)} chunks to vector store...")
        self.components['vector_store'].add_documents(all_chunks, embeddings)
        
        # Initialize hybrid search if needed
        if hasattr(self.hybrid_search, 'fit'):
            self.hybrid_search.fit(chunk_texts)
        
        logging.info("Document indexing complete.")
    
    def retrieve(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using modular components and advanced features.
        """
        logging.info(f"Retrieving top {k} documents for query: '{query}'")
        
        # Get query embedding
        query_embedding = self.components['embedder'].embed([query])[0]
        
        # Perform initial retrieval
        retrieved_docs = self.components['vector_store'].search(
            query_embedding, k=k*2, filters=filters  # Retrieve more for reranking
        )
        
        if not retrieved_docs:
            logging.warning("No documents retrieved")
            return []
        
        # Apply hybrid search if configured
        if hasattr(self.hybrid_search, 'search'):
            try:
                dense_scores = [1.0 - doc.get('distance', 0.0) for doc in retrieved_docs]
                hybrid_results = self.hybrid_search.search(query, dense_scores, k=k)
                
                # Update retrieved docs with hybrid scores
                for i, result in enumerate(hybrid_results):
                    if i < len(retrieved_docs):
                        retrieved_docs[i]['hybrid_score'] = result['hybrid_score']
            except Exception as e:
                logging.warning(f"Hybrid search failed: {e}")
        
        # Apply reranking
        if self.reranker:
            try:
                retrieved_docs = self.reranker.rerank(query, retrieved_docs)
            except Exception as e:
                logging.warning(f"Reranking failed: {e}")
        
        # Return top k results
        return retrieved_docs[:k]
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using modular components and context compression.
        """
        if not retrieved_docs:
            return "I am sorry, but I could not find enough relevant information to answer your question."
        
        # Extract context from retrieved documents
        context = [doc['content'] for doc in retrieved_docs]
        
        # Apply context compression if needed
        max_context_tokens = self.config.get('max_context_tokens', 1000)
        if self.context_compressor:
            try:
                context = self.context_compressor.compress(context, query, max_context_tokens)
            except Exception as e:
                logging.warning(f"Context compression failed: {e}")
        
        # Generate answer
        logging.info("Generating answer...")
        answer = self.components['generator'].generate_answer(query, context)
        logging.info("Answer generated.")
        
        return answer
    
    def query(self, question: str, k: int = 5, filters: Dict[str, Any] = None, 
              use_memory: bool = True) -> str:
        """
        End-to-end query processing with conversation memory.
        """
        # Add conversation context if using memory
        if use_memory and self.conversation_memory.conversation_history:
            context_summary = self.conversation_memory.get_summary_context()
            if context_summary:
                question = f"Previous conversation:\n{context_summary}\n\nCurrent question: {question}"
        
        # Retrieve documents
        retrieved_docs = self.retrieve(question, k=k, filters=filters)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        # Update conversation memory
        if use_memory:
            self.conversation_memory.add_turn(question, answer, retrieved_docs)
        
        return answer
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        """
        return self.conversation_memory.conversation_history
    
    def clear_conversation(self):
        """
        Clear conversation history.
        """
        self.conversation_memory.clear()
        logging.info("Conversation history cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration.
        """
        return {
            'config': self.config,
            'components': {
                'text_splitter': type(self.components['text_splitter']).__name__,
                'embedder': type(self.components['embedder']).__name__,
                'vector_store': type(self.components['vector_store']).__name__,
                'generator': type(self.components['generator']).__name__,
                'reranker': type(self.reranker).__name__ if self.reranker else None,
                'context_compressor': type(self.context_compressor).__name__,
                'hybrid_search': type(self.hybrid_search).__name__
            },
            'conversation_turns': len(self.conversation_memory.conversation_history)
        } 