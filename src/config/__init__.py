"""
Configuration module for RAG system

This module provides a hierarchical configuration system with:
- Environment-specific configurations (development, testing, production)
- Pydantic validation
- Environment variable overrides
- Secret management
- Backward compatibility

Usage:
    from src.config import get_config, get_config_manager
    
    # Get validated configuration
    config = get_config()
    
    # Access specific values
    debug_mode = config.system.debug
    llm_model = config.generator.model_name
    
    # Get configuration manager for advanced operations
    manager = get_config_manager()
"""

from .manager import (
    ConfigurationManager,
    ConfigurationError,
    get_config_manager,
    get_config,
    get_secrets
)
from .models import (
    RAGConfiguration,
    SecretConfig,
    SystemConfig,
    TextSplitterConfig,
    EmbedderConfig,
    VectorStoreConfig,
    GeneratorConfig,
    RetrievalConfig,
    ContextConfig,
    ConversationConfig,
    SecurityConfig,
    RateLimitingConfig,
    APIConfig,
    CORSConfig,
    LoggingConfig,
    MonitoringConfig,
    CacheConfig,
    EvaluationConfig,
    FeatureFlagsConfig,
    StorageConfig,
    ErrorHandlingConfig,
    PerformanceConfig,
    QueryOptimizationConfig,
    KnowledgeGraphConfig,
    ABTestingConfig,
    EnvironmentSpecificConfig
)

__all__ = [
    # Main interfaces
    'get_config',
    'get_config_manager',
    'get_secrets',
    
    # Manager and exceptions
    'ConfigurationManager',
    'ConfigurationError',
    
    # Configuration models
    'RAGConfiguration',
    'SecretConfig',
    'SystemConfig',
    'TextSplitterConfig',
    'EmbedderConfig',
    'VectorStoreConfig',
    'GeneratorConfig',
    'RetrievalConfig',
    'ContextConfig',
    'ConversationConfig',
    'SecurityConfig',
    'RateLimitingConfig',
    'APIConfig',
    'CORSConfig',
    'LoggingConfig',
    'MonitoringConfig',
    'CacheConfig',
    'EvaluationConfig',
    'FeatureFlagsConfig',
    'StorageConfig',
    'ErrorHandlingConfig',
    'PerformanceConfig',
    'QueryOptimizationConfig',
    'KnowledgeGraphConfig',
    'ABTestingConfig',
    'EnvironmentSpecificConfig'
]