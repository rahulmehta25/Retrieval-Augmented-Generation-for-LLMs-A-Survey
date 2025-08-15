"""
Pydantic models for configuration validation
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from pathlib import Path
import os


class SystemConfig(BaseModel):
    """System configuration settings"""
    name: str = "RAG System"
    version: str = "2.0.0"
    debug: bool = False
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout: int = Field(default=30, ge=1)
    cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, ge=0)


class TextSplitterConfig(BaseModel):
    """Text splitter configuration"""
    type: Literal["fixed_size", "semantic", "sliding", "sentence"] = "semantic"
    chunk_size: int = Field(default=512, ge=50, le=8192)
    chunk_overlap: int = Field(default=128, ge=0)
    min_chunk_size: int = Field(default=100, ge=10)
    max_chunk_size: int = Field(default=1000, ge=100)
    respect_sentences: bool = True
    respect_paragraphs: bool = True

    @validator('chunk_overlap')
    def overlap_less_than_size(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


class EmbedderConfig(BaseModel):
    """Embedding model configuration"""
    type: Literal["sentence_transformer", "openai", "cohere", "huggingface"] = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "./embedding_cache"
    dimension: int = Field(default=384, ge=1)
    batch_size: int = Field(default=32, ge=1, le=512)
    normalize: bool = True
    cache_embeddings: bool = True

    @validator('cache_dir')
    def validate_cache_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    type: Literal["chromadb", "faiss", "hybrid"] = "chromadb"
    path: str = "./chroma_db"
    collection_name: str = "rag_collection"
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = "cosine"
    index_type: Literal["HNSW", "IVF", "Flat"] = "HNSW"
    ef_construction: int = Field(default=200, ge=16)
    ef_search: int = Field(default=50, ge=1)

    @validator('path')
    def validate_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class GeneratorConfig(BaseModel):
    """LLM generator configuration"""
    type: Literal["ollama", "openai", "anthropic", "huggingface"] = "ollama"
    model_name: str = "gemma:2b"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: List[str] = Field(default_factory=lambda: ["\n\n"])
    host: str = "localhost"
    port: int = Field(default=11434, ge=1, le=65535)

    @validator('stop_sequences')
    def validate_stop_sequences(cls, v):
        if len(v) > 10:
            raise ValueError('Too many stop sequences')
        return v


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    strategy: Literal["dense", "sparse", "hybrid", "adaptive"] = "hybrid"
    k_documents: int = Field(default=10, ge=1, le=100)
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = Field(default=5, ge=1)
    use_mmr: bool = True
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator('rerank_top_k')
    def rerank_less_than_k(cls, v, values):
        if 'k_documents' in values and v > values['k_documents']:
            raise ValueError('rerank_top_k must be <= k_documents')
        return v


class ContextConfig(BaseModel):
    """Context management configuration"""
    max_context_length: int = Field(default=2048, ge=128)
    compression_enabled: bool = False
    compression_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
    summarization_enabled: bool = False
    context_ordering: Literal["relevance", "diversity", "recency"] = "relevance"


class ConversationConfig(BaseModel):
    """Conversation management configuration"""
    memory_enabled: bool = True
    memory_type: Literal["buffer", "summary", "knowledge_graph"] = "buffer"
    max_history: int = Field(default=10, ge=0, le=100)
    max_turns: int = Field(default=10, ge=0, le=100)
    max_tokens: int = Field(default=2000, ge=100)
    summarize_after: int = Field(default=5, ge=1)
    context_window: int = Field(default=3, ge=1)


class SecurityConfig(BaseModel):
    """Security configuration"""
    input_validation: bool = True
    max_input_length: int = Field(default=1000, ge=1)
    max_query_length: int = Field(default=1000, ge=1)
    max_document_size: int = Field(default=10485760, ge=1024)  # 10MB
    sanitize_outputs: bool = True
    block_harmful_content: bool = True
    audit_logging: bool = False


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration"""
    enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=1000, ge=1)
    burst_size: int = Field(default=10, ge=1)
    strategy: Literal["sliding_window", "fixed_window", "token_bucket"] = "sliding_window"
    persistent_storage: Optional[str] = None


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = "localhost"
    port: int = Field(default=8000, ge=1, le=65535)
    base_path: str = "/api"
    version: str = "v1"
    workers: int = Field(default=4, ge=1, le=32)
    timeout: int = Field(default=30, ge=1)
    cors_enabled: bool = True


class CORSConfig(BaseModel):
    """CORS configuration"""
    enabled: bool = True
    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allowed_methods: List[str] = Field(default_factory=lambda: ["GET", "POST"])
    allowed_headers: List[str] = Field(default_factory=lambda: ["Content-Type", "Authorization"])
    expose_headers: List[str] = Field(default_factory=list)
    allow_credentials: bool = True
    max_age: int = Field(default=600, ge=0)


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "simple", "detailed"] = "json"
    console_enabled: bool = True
    file_enabled: bool = False
    max_file_size: int = Field(default=104857600, ge=1024)  # 100MB
    backup_count: int = Field(default=5, ge=0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = False
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    performance_tracking: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class CacheConfig(BaseModel):
    """Cache configuration"""
    embedding_cache: bool = True
    query_cache: bool = True
    response_cache: bool = False
    cache_backend: Literal["redis", "memcached", "in_memory"] = "in_memory"
    redis_url: Optional[str] = None
    ttl_seconds: int = Field(default=3600, ge=0)
    max_size: int = Field(default=1000, ge=1)

    @validator('redis_url')
    def validate_redis_url(cls, v, values):
        if values.get('cache_backend') == 'redis' and not v:
            raise ValueError('redis_url is required when using redis backend')
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    enabled: bool = False
    ragas_enabled: bool = False
    ragas_metrics: List[str] = Field(default_factory=lambda: ["faithfulness", "answer_relevancy"])
    min_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    log_evaluations: bool = False
    auto_evaluate: bool = False
    evaluation_interval: int = Field(default=10, ge=1)


class FeatureFlagsConfig(BaseModel):
    """Feature flags configuration"""
    streaming_responses: bool = False
    multi_language_support: bool = False
    voice_input: bool = False
    export_conversations: bool = False
    collaborative_filtering: bool = False
    active_learning: bool = False
    federated_search: bool = False
    knowledge_graph_enabled: bool = False
    ab_testing_enabled: bool = False


class StorageConfig(BaseModel):
    """Storage configuration"""
    base_path: str = "./rag_storage"
    backup_enabled: bool = False
    backup_interval_hours: int = Field(default=24, ge=1)
    backup_retention_days: int = Field(default=7, ge=1)

    @validator('base_path')
    def validate_base_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1, ge=0)
    fallback_to_simple: bool = True
    log_errors: bool = True
    alert_on_failure: bool = False


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    batch_processing: bool = True
    batch_size: int = Field(default=32, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1)
    use_gpu: bool = False
    gpu_device_id: int = Field(default=0, ge=0)
    memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)


class QueryOptimizationConfig(BaseModel):
    """Query optimization configuration"""
    enabled: bool = True
    rewrite: bool = True
    expand: bool = True
    decompose: bool = True
    use_hyde: bool = False
    max_expansion: int = Field(default=3, ge=1, le=10)
    complexity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph configuration"""
    enabled: bool = False
    max_entities_per_doc: int = Field(default=50, ge=1)
    max_relations_per_doc: int = Field(default=100, ge=1)
    max_hops: int = Field(default=2, ge=1, le=5)
    max_nodes_per_query: int = Field(default=20, ge=1)


class ABTestingConfig(BaseModel):
    """A/B testing configuration"""
    enabled: bool = False
    default_allocation: Literal["random", "weighted", "deterministic"] = "random"
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1)
    minimum_sample_size: int = Field(default=100, ge=10)


class EnvironmentSpecificConfig(BaseModel):
    """Environment-specific configuration that varies by env"""
    environment: Literal["development", "testing", "staging", "production"] = "development"
    hot_reload: bool = False
    verbose_logging: bool = False
    mock_llm: bool = False
    save_intermediate_results: bool = False
    profiling_enabled: bool = False


class RAGConfiguration(BaseModel):
    """Main RAG system configuration"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    query_optimization: QueryOptimizationConfig = Field(default_factory=QueryOptimizationConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    ab_testing: ABTestingConfig = Field(default_factory=ABTestingConfig)
    environment_specific: EnvironmentSpecificConfig = Field(default_factory=EnvironmentSpecificConfig)

    class Config:
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment


# Secret configuration that should come from environment variables
class SecretConfig(BaseModel):
    """Configuration for secrets that should come from environment variables"""
    # Database
    database_url: Optional[str] = Field(default=None, env="RAG_DATABASE_URL")
    
    # Redis
    redis_url: Optional[str] = Field(default=None, env="RAG_REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="RAG_REDIS_PASSWORD")
    
    # JWT
    jwt_secret_key: Optional[str] = Field(default=None, env="RAG_JWT_SECRET_KEY")
    
    # SSL
    ssl_keyfile: Optional[str] = Field(default=None, env="RAG_SSL_KEYFILE")
    ssl_certfile: Optional[str] = Field(default=None, env="RAG_SSL_CERTFILE")
    
    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # Neo4j
    neo4j_url: Optional[str] = Field(default=None, env="RAG_NEO4J_URL")
    neo4j_user: Optional[str] = Field(default=None, env="RAG_NEO4J_USER")
    neo4j_password: Optional[str] = Field(default=None, env="RAG_NEO4J_PASSWORD")
    
    # Monitoring
    alert_webhook_url: Optional[str] = Field(default=None, env="RAG_ALERT_WEBHOOK")
    
    # SMTP
    smtp_server: Optional[str] = Field(default=None, env="RAG_SMTP_SERVER")
    smtp_username: Optional[str] = Field(default=None, env="RAG_SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="RAG_SMTP_PASSWORD")
    
    # Encryption
    encryption_key: Optional[str] = Field(default=None, env="RAG_ENCRYPTION_KEY")
    kdf_salt: Optional[str] = Field(default=None, env="RAG_KDF_SALT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"