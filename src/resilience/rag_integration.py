"""
RAG System Resilience Integration

Integrates resilience patterns into the RAG system components.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .retry_policy import RetryPolicy, RetryStrategy
from .bulkhead import SemaphoreBulkhead, ThreadPoolBulkhead
from .timeout_manager import TimeoutManager, AdaptiveTimeoutManager
from .fallback import FallbackHandler, FallbackStrategy
from .health_checker import HealthChecker, HealthCheck, HealthStatus
from .monitoring import ResilienceMonitor, MetricsCollector
from .tracing import DistributedTracer, TraceContext

logger = logging.getLogger(__name__)


class ResilientLLMClient:
    """
    Resilient wrapper for LLM API clients (OpenAI, Anthropic, etc.)
    
    Adds circuit breakers, retries, timeouts, and fallbacks to LLM calls.
    """
    
    def __init__(
        self,
        base_client: Any,
        service_name: str = "llm",
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        enable_timeout: bool = True,
        enable_fallback: bool = True,
        enable_tracing: bool = True
    ):
        """
        Initialize resilient LLM client
        
        Args:
            base_client: Original LLM client
            service_name: Name for the service
            enable_circuit_breaker: Enable circuit breaker
            enable_retry: Enable retry logic
            enable_timeout: Enable timeout management
            enable_fallback: Enable fallback mechanism
            enable_tracing: Enable distributed tracing
        """
        self.base_client = base_client
        self.service_name = service_name
        
        # Initialize resilience components
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                name=f"{service_name}_circuit_breaker",
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=Exception,
                failure_rate_threshold=0.5,
                minimum_number_of_calls=10
            )
        else:
            self.circuit_breaker = None
        
        if enable_retry:
            self.retry_policy = RetryPolicy(
                max_attempts=3,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                initial_delay=1.0,
                max_delay=30.0,
                jitter=True,
                retriable_exceptions=[
                    ConnectionError,
                    TimeoutError,
                    IOError
                ]
            )
        else:
            self.retry_policy = None
        
        if enable_timeout:
            self.timeout_manager = AdaptiveTimeoutManager(
                initial_timeout=30.0,
                min_timeout=5.0,
                max_timeout=120.0
            )
        else:
            self.timeout_manager = None
        
        if enable_fallback:
            self.fallback_handler = FallbackHandler(
                strategy=FallbackStrategy.CACHED_VALUE,
                cache_ttl=300.0,
                fallback_value="I'm currently unable to process your request. Please try again later."
            )
        else:
            self.fallback_handler = None
        
        if enable_tracing:
            self.tracer = DistributedTracer(
                service_name=service_name,
                enabled=True,
                sampling_rate=0.1
            )
        else:
            self.tracer = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text with resilience patterns
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Start tracing
        if self.tracer:
            trace_context = self.tracer.start_span(
                "llm_generate",
                tags={"prompt_length": len(prompt)}
            )
        else:
            trace_context = None
        
        try:
            # Define the core function
            def _generate():
                return self.base_client.generate(prompt, **kwargs)
            
            # Apply timeout
            if self.timeout_manager:
                _generate = lambda: self.timeout_manager.execute(_generate)
            
            # Apply retry
            if self.retry_policy:
                _generate = lambda: self.retry_policy.execute(_generate)
            
            # Apply circuit breaker
            if self.circuit_breaker:
                _generate = lambda: self.circuit_breaker.call(_generate)
            
            # Apply fallback
            if self.fallback_handler:
                result = self.fallback_handler.execute(
                    _generate,
                    cache_key=f"llm_{prompt[:50]}_{kwargs}"
                )
            else:
                result = _generate()
            
            # Finish tracing
            if self.tracer and trace_context:
                self.tracer.finish_span(trace_context, status="success")
            
            return result
            
        except Exception as e:
            # Finish tracing with error
            if self.tracer and trace_context:
                self.tracer.finish_span(trace_context, status="error", error=str(e))
            raise
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        Async generate text with resilience patterns
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Start tracing
        if self.tracer:
            trace_context = self.tracer.start_span(
                "llm_generate_async",
                tags={"prompt_length": len(prompt)}
            )
        else:
            trace_context = None
        
        try:
            # Define the core function
            async def _generate():
                return await self.base_client.generate_async(prompt, **kwargs)
            
            # Apply timeout
            if self.timeout_manager:
                _generate = lambda: self.timeout_manager.execute_async(_generate)
            
            # Apply retry
            if self.retry_policy:
                _generate = lambda: self.retry_policy.execute_async(_generate)
            
            # Apply circuit breaker
            if self.circuit_breaker:
                _generate = lambda: self.circuit_breaker.call_async(_generate)
            
            # Apply fallback
            if self.fallback_handler:
                result = await self.fallback_handler.execute_async(
                    _generate,
                    cache_key=f"llm_{prompt[:50]}_{kwargs}"
                )
            else:
                result = await _generate()
            
            # Finish tracing
            if self.tracer and trace_context:
                self.tracer.finish_span(trace_context, status="success")
            
            return result
            
        except Exception as e:
            # Finish tracing with error
            if self.tracer and trace_context:
                self.tracer.finish_span(trace_context, status="error", error=str(e))
            raise


class ResilientVectorStore:
    """
    Resilient wrapper for vector database operations
    
    Adds resilience patterns to vector store operations.
    """
    
    def __init__(
        self,
        base_store: Any,
        service_name: str = "vector_store",
        max_concurrent_queries: int = 50,
        enable_bulkhead: bool = True
    ):
        """
        Initialize resilient vector store
        
        Args:
            base_store: Original vector store
            service_name: Name for the service
            max_concurrent_queries: Maximum concurrent queries
            enable_bulkhead: Enable bulkhead pattern
        """
        self.base_store = base_store
        self.service_name = service_name
        
        # Circuit breaker for vector store
        self.circuit_breaker = CircuitBreaker(
            name=f"{service_name}_circuit_breaker",
            failure_threshold=10,
            recovery_timeout=30.0,
            failure_rate_threshold=0.3
        )
        
        # Retry policy for transient failures
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=0.5,
            max_delay=10.0
        )
        
        # Bulkhead for resource isolation
        if enable_bulkhead:
            self.bulkhead = SemaphoreBulkhead(
                name=f"{service_name}_bulkhead",
                max_concurrent=max_concurrent_queries,
                max_wait_time=5.0
            )
        else:
            self.bulkhead = None
        
        # Timeout manager
        self.timeout_manager = TimeoutManager(
            default_timeout=10.0
        )
        
        # Fallback for search operations
        self.fallback_handler = FallbackHandler(
            strategy=FallbackStrategy.CACHED_VALUE,
            cache_ttl=60.0,
            fallback_value=[]
        )
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with resilience patterns
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filters: Optional filters
            
        Returns:
            Search results
        """
        def _search():
            # Apply bulkhead if enabled
            if self.bulkhead:
                return self.bulkhead.execute(
                    lambda: self.base_store.search(query_embedding, k, filters)
                )
            else:
                return self.base_store.search(query_embedding, k, filters)
        
        # Apply timeout
        _search_with_timeout = lambda: self.timeout_manager.execute(_search)
        
        # Apply retry
        _search_with_retry = lambda: self.retry_policy.execute(_search_with_timeout)
        
        # Apply circuit breaker
        _search_with_breaker = lambda: self.circuit_breaker.call(_search_with_retry)
        
        # Apply fallback
        return self.fallback_handler.execute(
            _search_with_breaker,
            cache_key=f"search_{hash(tuple(query_embedding[:5]))}_{k}_{filters}"
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """
        Add documents with resilience patterns
        
        Args:
            documents: Documents to add
            embeddings: Document embeddings
        """
        def _add():
            return self.base_store.add_documents(documents, embeddings)
        
        # Apply retry for write operations
        _add_with_retry = lambda: self.retry_policy.execute(_add)
        
        # Apply circuit breaker
        return self.circuit_breaker.call(_add_with_retry)


class ResilientRAGOrchestrator:
    """
    Orchestrates resilient RAG operations
    
    Coordinates resilient components for end-to-end RAG pipeline.
    """
    
    def __init__(
        self,
        llm_client: ResilientLLMClient,
        vector_store: ResilientVectorStore,
        enable_monitoring: bool = True
    ):
        """
        Initialize resilient RAG orchestrator
        
        Args:
            llm_client: Resilient LLM client
            vector_store: Resilient vector store
            enable_monitoring: Enable monitoring
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        
        # Health checker for all components
        self.health_checker = HealthChecker(
            check_interval=30.0,
            enable_auto_recovery=True
        )
        
        # Register health checks
        self._register_health_checks()
        
        # Monitoring
        if enable_monitoring:
            self.metrics_collector = MetricsCollector()
            self.monitor = ResilienceMonitor(self.metrics_collector)
            
            # Register components for monitoring
            self._register_monitoring()
            
            # Start monitoring
            self.metrics_collector.start()
            self.health_checker.start()
    
    def _register_health_checks(self):
        """Register health checks for all components"""
        
        # LLM health check
        def check_llm():
            try:
                # Simple health check - try to generate with minimal prompt
                result = self.llm_client.generate("Hello", max_tokens=1)
                return bool(result)
            except:
                return False
        
        self.health_checker.register_check(
            HealthCheck(
                name="llm_service",
                check_function=check_llm,
                interval=60.0,
                failure_threshold=3,
                critical=True
            )
        )
        
        # Vector store health check
        def check_vector_store():
            try:
                # Try a simple search
                result = self.vector_store.search([0.0] * 100, k=1)
                return True
            except:
                return False
        
        self.health_checker.register_check(
            HealthCheck(
                name="vector_store",
                check_function=check_vector_store,
                interval=30.0,
                failure_threshold=3,
                critical=True
            )
        )
    
    def _register_monitoring(self):
        """Register components for monitoring"""
        if self.llm_client.circuit_breaker:
            self.metrics_collector.register_circuit_breaker(
                "llm", self.llm_client.circuit_breaker
            )
        
        if self.vector_store.circuit_breaker:
            self.metrics_collector.register_circuit_breaker(
                "vector_store", self.vector_store.circuit_breaker
            )
        
        if self.vector_store.bulkhead:
            self.metrics_collector.register_bulkhead(
                "vector_store", self.vector_store.bulkhead
            )
    
    def query(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute RAG query with full resilience
        
        Args:
            query: User query
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            RAG response with context and answer
        """
        # Check overall health
        if self.health_checker.get_status() == HealthStatus.UNHEALTHY:
            logger.warning("System unhealthy, using degraded mode")
            return {
                'answer': "System is currently experiencing issues. Please try again later.",
                'context': [],
                'status': 'degraded'
            }
        
        try:
            # Retrieve relevant documents
            # (Embedding generation would also be wrapped in resilience patterns)
            query_embedding = self._generate_embedding(query)
            documents = self.vector_store.search(query_embedding, k=k)
            
            # Generate answer
            context = "\n".join([doc.get('content', '') for doc in documents])
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            answer = self.llm_client.generate(prompt, **kwargs)
            
            return {
                'answer': answer,
                'context': documents,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                'answer': "Unable to process your query at this time.",
                'context': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with resilience (simplified)"""
        # This would typically call an embedding service with resilience
        # For demonstration, returning dummy embedding
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val >> i) & 1 for i in range(768)]  # Dummy 768-dim embedding
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        return {
            'overall': self.health_checker.get_status().value,
            'components': self.health_checker.get_all_results(),
            'metrics': self.metrics_collector.get_latest_metrics() if hasattr(self, 'metrics_collector') else None
        }
    
    def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        if hasattr(self, 'health_checker'):
            self.health_checker.stop()
        
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()