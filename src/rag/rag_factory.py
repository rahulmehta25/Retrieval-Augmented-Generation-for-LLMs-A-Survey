"""
RAG Component Factory for Modular Architecture

This module implements a factory pattern for creating RAG components dynamically
based on configuration, enabling a modular and pluggable architecture.
"""

from typing import Dict, Any, Optional
from src.chunking.text_splitter import FixedSizeTextSplitter, SentenceTextSplitter
from src.embedding.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import ChromaDBVectorStore, FAISSVectorStore
from src.generation.generator import HuggingFaceGenerator, OpenAIGenerator
from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker

class RAGComponentFactory:
    """
    Factory class for creating RAG components based on configuration.
    This enables a modular architecture where components can be swapped
    without changing the core pipeline.
    """
    
    @staticmethod
    def get_text_splitter(config: dict) -> Any:
        """
        Creates a text splitter based on configuration.
        
        Args:
            config: Configuration dictionary for the text splitter
            
        Returns:
            Configured text splitter instance
            
        Raises:
            ValueError: If the splitter type is unknown
        """
        splitter_type = config.get("type", "fixed_size")
        
        if splitter_type == "fixed_size":
            return FixedSizeTextSplitter(
                chunk_size=config.get("chunk_size", 500),
                chunk_overlap=config.get("chunk_overlap", 50)
            )
        elif splitter_type == "sentence":
            return SentenceTextSplitter()
        else:
            raise ValueError(f"Unknown text splitter type: {splitter_type}")

    @staticmethod
    def get_embedder(config: dict) -> Any:
        """
        Creates an embedder based on configuration.
        
        Args:
            config: Configuration dictionary for the embedder
            
        Returns:
            Configured embedder instance
            
        Raises:
            ValueError: If the embedder type is unknown
        """
        embedder_type = config.get("type", "sentence_transformer")
        
        if embedder_type == "sentence_transformer":
            return SentenceTransformerEmbedder(
                model_name=config.get("model_name", "all-MiniLM-L6-v2"),
                cache_dir=config.get("cache_dir", "./embedding_cache")
            )
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    @staticmethod
    def get_vector_store(config: dict, embedder_instance: Optional[Any] = None) -> Any:
        """
        Creates a vector store based on configuration.
        
        Args:
            config: Configuration dictionary for the vector store
            embedder_instance: Optional embedder instance for internal use
            
        Returns:
            Configured vector store instance
            
        Raises:
            ValueError: If the vector store type is unknown
        """
        store_type = config.get("type", "chromadb")
        
        if store_type == "chromadb":
            return ChromaDBVectorStore(
                path=config.get("path", "./chroma_db"),
                collection_name=config.get("collection_name", "rag_collection")
            )
        elif store_type == "faiss":
            embedding_dimension = config.get("embedding_dimension", 384)
            return FAISSVectorStore(embedding_dimension=embedding_dimension)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def get_generator(config: dict) -> Any:
        """
        Creates a generator based on configuration.
        
        Args:
            config: Configuration dictionary for the generator
            
        Returns:
            Configured generator instance
            
        Raises:
            ValueError: If the generator type is unknown
        """
        generator_type = config.get("type", "huggingface")
        
        if generator_type == "huggingface":
            return HuggingFaceGenerator(
                model_name=config.get("model_name", "distilgpt2"),
                device=config.get("device", "cpu")
            )
        elif generator_type == "openai":
            return OpenAIGenerator(
                model_name=config.get("model_name", "gpt-3.5-turbo")
            )
        elif generator_type == "ollama":
            from src.generation.generator import OllamaGenerator
            return OllamaGenerator(
                model_name=config.get("model_name", "gemma:2b"),
                host=config.get("host", "localhost"),
                port=config.get("port", 11434)
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    @staticmethod
    def get_reranker(config: dict) -> Any:
        """
        Creates a reranker based on configuration.
        
        Args:
            config: Configuration dictionary for the reranker
            
        Returns:
            Configured reranker instance
            
        Raises:
            ValueError: If the reranker type is unknown
        """
        reranker_type = config.get("type", "cross_encoder")
        
        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(
                model_name=config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
        elif reranker_type == "simple":
            return SimpleReranker()
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    @staticmethod
    def create_rag_system(config: dict) -> Dict[str, Any]:
        """
        Creates a complete RAG system with all components based on configuration.
        
        Args:
            config: Complete configuration dictionary
            
        Returns:
            Dictionary containing all RAG components
            
        Raises:
            ValueError: If any component configuration is invalid
        """
        try:
            # Create embedder first as it might be needed by vector store
            embedder = RAGComponentFactory.get_embedder(config.get("embedder", {}))
            
            # Create other components
            text_splitter = RAGComponentFactory.get_text_splitter(config.get("text_splitter", {}))
            vector_store = RAGComponentFactory.get_vector_store(config.get("vector_store", {}))
            generator = RAGComponentFactory.get_generator(config.get("generator", {}))
            
            # Create reranker if specified
            reranker = None
            if "reranker" in config:
                reranker = RAGComponentFactory.get_reranker(config["reranker"])
            
            return {
                "text_splitter": text_splitter,
                "embedder": embedder,
                "vector_store": vector_store,
                "generator": generator,
                "reranker": reranker
            }
            
        except Exception as e:
            raise ValueError(f"Failed to create RAG system: {str(e)}") 