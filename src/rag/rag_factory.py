"""
RAG Component Factory for Modular Architecture

This module implements a factory pattern for creating RAG components dynamically
based on configuration, enabling a modular and pluggable architecture.
"""

from typing import Dict, Any, Optional, Union
from src.chunking.text_splitter import FixedSizeTextSplitter, SentenceTextSplitter
from src.embedding.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import ChromaDBVectorStore, FAISSVectorStore
from src.generation.generator import HuggingFaceGenerator, OpenAIGenerator
from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker
from src.config import get_config, RAGConfiguration
from src.config.models import (
    TextSplitterConfig, EmbedderConfig, VectorStoreConfig, 
    GeneratorConfig, RetrievalConfig
)

class RAGComponentFactory:
    """
    Factory class for creating RAG components based on configuration.
    This enables a modular architecture where components can be swapped
    without changing the core pipeline.
    """
    
    @staticmethod
    def get_text_splitter(config: Union[dict, TextSplitterConfig, None] = None) -> Any:
        """
        Creates a text splitter based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            
        Returns:
            Configured text splitter instance
            
        Raises:
            ValueError: If the splitter type is unknown
        """
        # Handle different config input types
        if config is None:
            config = get_config().text_splitter
        elif isinstance(config, dict):
            # Backward compatibility with dict config
            config = TextSplitterConfig(**config)
        
        if config.type == "fixed_size":
            return FixedSizeTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        elif config.type == "sentence":
            return SentenceTextSplitter()
        elif config.type == "semantic":
            # Semantic splitter (if implemented)
            return SentenceTextSplitter()  # Fallback for now
        else:
            raise ValueError(f"Unknown text splitter type: {config.type}")

    @staticmethod
    def get_embedder(config: Union[dict, EmbedderConfig, None] = None) -> Any:
        """
        Creates an embedder based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            
        Returns:
            Configured embedder instance
            
        Raises:
            ValueError: If the embedder type is unknown
        """
        # Handle different config input types
        if config is None:
            config = get_config().embedder
        elif isinstance(config, dict):
            # Backward compatibility with dict config
            config = EmbedderConfig(**config)
        
        if config.type == "sentence_transformer":
            return SentenceTransformerEmbedder(
                model_name=config.model_name,
                cache_dir=config.cache_dir
            )
        else:
            raise ValueError(f"Unknown embedder type: {config.type}")

    @staticmethod
    def get_vector_store(config: Union[dict, VectorStoreConfig, None] = None, 
                        embedder_instance: Optional[Any] = None) -> Any:
        """
        Creates a vector store based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            embedder_instance: Optional embedder instance for internal use
            
        Returns:
            Configured vector store instance
            
        Raises:
            ValueError: If the vector store type is unknown
        """
        # Handle different config input types
        if config is None:
            config = get_config().vector_store
        elif isinstance(config, dict):
            # Backward compatibility with dict config
            config = VectorStoreConfig(**config)
        
        if config.type == "chromadb":
            return ChromaDBVectorStore(
                path=config.path,
                collection_name=config.collection_name
            )
        elif config.type == "faiss":
            # Get embedding dimension from embedder config
            embedder_config = get_config().embedder
            return FAISSVectorStore(embedding_dimension=embedder_config.dimension)
        elif config.type == "hybrid":
            # Return ChromaDB for now, hybrid implementation would be more complex
            return ChromaDBVectorStore(
                path=config.path,
                collection_name=config.collection_name
            )
        else:
            raise ValueError(f"Unknown vector store type: {config.type}")

    @staticmethod
    def get_generator(config: Union[dict, GeneratorConfig, None] = None) -> Any:
        """
        Creates a generator based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            
        Returns:
            Configured generator instance
            
        Raises:
            ValueError: If the generator type is unknown
        """
        # Handle different config input types
        if config is None:
            config = get_config().generator
        elif isinstance(config, dict):
            # Backward compatibility with dict config
            config = GeneratorConfig(**config)
        
        if config.type == "huggingface":
            return HuggingFaceGenerator(
                model_name=config.model_name,
                device="cpu"  # Default device
            )
        elif config.type == "openai":
            return OpenAIGenerator(
                model_name=config.model_name
            )
        elif config.type == "ollama":
            from src.generation.generator import OllamaGenerator
            return OllamaGenerator(
                model_name=config.model_name,
                host=config.host,
                port=config.port
            )
        else:
            raise ValueError(f"Unknown generator type: {config.type}")

    @staticmethod
    def get_reranker(config: Union[dict, RetrievalConfig, None] = None) -> Any:
        """
        Creates a reranker based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            
        Returns:
            Configured reranker instance
            
        Raises:
            ValueError: If the reranker type is unknown
        """
        # Handle different config input types
        if config is None:
            retrieval_config = get_config().retrieval
            if not retrieval_config.rerank:
                return None
            model_name = retrieval_config.reranker_model
            reranker_type = "cross_encoder"
        elif isinstance(config, dict):
            reranker_type = config.get("type", "cross_encoder")
            model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:  # RetrievalConfig
            if not config.rerank:
                return None
            model_name = config.reranker_model
            reranker_type = "cross_encoder"
        
        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(model_name=model_name)
        elif reranker_type == "simple":
            return SimpleReranker()
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    @staticmethod
    def create_rag_system(config: Union[dict, RAGConfiguration, None] = None) -> Dict[str, Any]:
        """
        Creates a complete RAG system with all components based on configuration.
        
        Args:
            config: Configuration object, dict, or None (uses global config)
            
        Returns:
            Dictionary containing all RAG components
            
        Raises:
            ValueError: If any component configuration is invalid
        """
        try:
            # Handle different config input types
            if config is None:
                config = get_config()
            elif isinstance(config, dict):
                # Legacy support for dict config
                embedder = RAGComponentFactory.get_embedder(config.get("embedder", {}))
                text_splitter = RAGComponentFactory.get_text_splitter(config.get("text_splitter", {}))
                vector_store = RAGComponentFactory.get_vector_store(config.get("vector_store", {}))
                generator = RAGComponentFactory.get_generator(config.get("generator", {}))
                
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
            
            # Use the new configuration system
            embedder = RAGComponentFactory.get_embedder(config.embedder)
            text_splitter = RAGComponentFactory.get_text_splitter(config.text_splitter)
            vector_store = RAGComponentFactory.get_vector_store(config.vector_store)
            generator = RAGComponentFactory.get_generator(config.generator)
            reranker = RAGComponentFactory.get_reranker(config.retrieval)
            
            return {
                "text_splitter": text_splitter,
                "embedder": embedder,
                "vector_store": vector_store,
                "generator": generator,
                "reranker": reranker,
                "config": config  # Include config for reference
            }
            
        except Exception as e:
            raise ValueError(f"Failed to create RAG system: {str(e)}")
    
    @staticmethod
    def create_rag_system_from_environment(environment: str = "development") -> Dict[str, Any]:
        """
        Creates a RAG system configured for a specific environment
        
        Args:
            environment: Environment name (development, testing, production)
            
        Returns:
            Dictionary containing all RAG components configured for the environment
        """
        from src.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager(environment=environment)
        config = config_manager.get_config()
        
        return RAGComponentFactory.create_rag_system(config) 