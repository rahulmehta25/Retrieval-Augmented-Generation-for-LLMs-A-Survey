"""
Example: Resilient RAG System

Demonstrates how to integrate resilience patterns into a production RAG system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import resilience components
from src.resilience import (
    CircuitBreaker,
    RetryPolicy,
    RetryStrategy,
    SemaphoreBulkhead,
    TimeoutManager,
    FallbackHandler,
    FallbackStrategy,
    HealthChecker,
    HealthCheck,
    ResilienceMonitor,
    MetricsCollector,
    ChaosEngine,
    ChaosStrategy,
    ChaosConfig,
    DistributedTracer
)
from src.resilience.rag_integration import (
    ResilientLLMClient,
    ResilientVectorStore,
    ResilientRAGOrchestrator
)


class MockLLMClient:
    """Mock LLM client for demonstration"""
    
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text (with simulated failures)"""
        self.call_count += 1
        
        # Simulate occasional failures
        import random
        if random.random() < self.failure_rate:
            raise ConnectionError("LLM service temporarily unavailable")
        
        # Simulate latency
        time.sleep(random.uniform(0.1, 0.5))
        
        return f"Generated response for: {prompt[:50]}..."
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generate text"""
        self.call_count += 1
        
        # Simulate occasional failures
        import random
        if random.random() < self.failure_rate:
            raise ConnectionError("LLM service temporarily unavailable")
        
        # Simulate latency
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return f"Async generated response for: {prompt[:50]}..."


class MockVectorStore:
    """Mock vector store for demonstration"""
    
    def __init__(self, failure_rate: float = 0.05):
        self.failure_rate = failure_rate
        self.documents = []
        self.embeddings = []
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        import random
        if random.random() < self.failure_rate:
            raise IOError("Vector database connection failed")
        
        # Return mock results
        return [
            {
                'id': f'doc_{i}',
                'content': f'Document {i} content relevant to query',
                'score': 0.9 - (i * 0.1)
            }
            for i in range(min(k, 3))
        ]
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """Add documents to store"""
        import random
        if random.random() < self.failure_rate:
            raise IOError("Vector database write failed")
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)


def demonstrate_basic_resilience():
    """Demonstrate basic resilience patterns"""
    logger.info("=== Basic Resilience Patterns Demo ===")
    
    # 1. Circuit Breaker
    logger.info("\n1. Circuit Breaker Pattern")
    circuit_breaker = CircuitBreaker(
        name="api_circuit",
        failure_threshold=3,
        recovery_timeout=5.0
    )
    
    def unreliable_api_call():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("API failed")
        return "Success"
    
    for i in range(5):
        try:
            result = circuit_breaker.call(unreliable_api_call)
            logger.info(f"  Call {i+1}: {result}")
        except Exception as e:
            logger.warning(f"  Call {i+1} failed: {e}")
        time.sleep(0.5)
    
    logger.info(f"  Circuit state: {circuit_breaker.state.value}")
    
    # 2. Retry Policy
    logger.info("\n2. Retry Policy with Exponential Backoff")
    retry_policy = RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=0.5
    )
    
    call_count = 0
    def flaky_service():
        nonlocal call_count
        call_count += 1
        logger.info(f"  Attempt {call_count}")
        if call_count < 3:
            raise ConnectionError("Service unavailable")
        return "Finally succeeded!"
    
    try:
        result = retry_policy.execute(flaky_service)
        logger.info(f"  Result: {result}")
    except Exception as e:
        logger.error(f"  All retries failed: {e}")
    
    # 3. Bulkhead Pattern
    logger.info("\n3. Bulkhead Pattern for Resource Isolation")
    bulkhead = SemaphoreBulkhead(
        name="db_bulkhead",
        max_concurrent=2,
        max_wait_time=1.0
    )
    
    def database_query(query_id):
        logger.info(f"  Executing query {query_id}")
        time.sleep(0.5)
        return f"Result {query_id}"
    
    import threading
    threads = []
    for i in range(4):
        thread = threading.Thread(
            target=lambda id=i: bulkhead.execute(database_query, id)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # 4. Timeout Management
    logger.info("\n4. Timeout Management")
    timeout_manager = TimeoutManager(default_timeout=1.0)
    
    def slow_operation():
        time.sleep(0.5)
        return "Completed in time"
    
    def very_slow_operation():
        time.sleep(2.0)
        return "Too slow"
    
    try:
        result = timeout_manager.execute(slow_operation)
        logger.info(f"  Fast operation: {result}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
    
    try:
        result = timeout_manager.execute(very_slow_operation)
        logger.info(f"  Slow operation: {result}")
    except Exception as e:
        logger.error(f"  Timeout: {e}")
    
    # 5. Fallback Handler
    logger.info("\n5. Fallback Mechanism")
    fallback_handler = FallbackHandler(
        strategy=FallbackStrategy.STATIC_VALUE,
        fallback_value="Fallback response"
    )
    
    def failing_service():
        raise Exception("Primary service failed")
    
    result = fallback_handler.execute(failing_service)
    logger.info(f"  Result with fallback: {result}")


def demonstrate_resilient_rag():
    """Demonstrate resilient RAG system"""
    logger.info("\n=== Resilient RAG System Demo ===")
    
    # Create mock services
    mock_llm = MockLLMClient(failure_rate=0.2)
    mock_vector_store = MockVectorStore(failure_rate=0.1)
    
    # Wrap with resilience
    resilient_llm = ResilientLLMClient(
        base_client=mock_llm,
        service_name="llm_service",
        enable_circuit_breaker=True,
        enable_retry=True,
        enable_timeout=True,
        enable_fallback=True
    )
    
    resilient_vector_store = ResilientVectorStore(
        base_store=mock_vector_store,
        service_name="vector_store",
        max_concurrent_queries=10
    )
    
    # Create orchestrator
    orchestrator = ResilientRAGOrchestrator(
        llm_client=resilient_llm,
        vector_store=resilient_vector_store,
        enable_monitoring=True
    )
    
    # Simulate queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?",
        "What are embeddings?",
        "Describe transformers"
    ]
    
    logger.info("\nProcessing queries with resilience:")
    for query in queries:
        logger.info(f"\nQuery: {query}")
        result = orchestrator.query(query, k=3)
        logger.info(f"  Status: {result['status']}")
        if result['status'] == 'success':
            logger.info(f"  Answer: {result['answer'][:100]}...")
            logger.info(f"  Context docs: {len(result['context'])}")
        else:
            logger.warning(f"  Error: {result.get('error', 'Unknown')}")
        
        time.sleep(0.5)
    
    # Check health status
    health_status = orchestrator.get_health_status()
    logger.info(f"\nSystem Health: {health_status['overall']}")
    
    # Get metrics
    if orchestrator.metrics_collector:
        metrics = orchestrator.metrics_collector.get_latest_metrics()
        if metrics:
            logger.info("\nResilience Metrics:")
            for cb_name, cb_metrics in metrics.circuit_breakers.items():
                logger.info(f"  {cb_name}: state={cb_metrics.get('state')}, "
                          f"failures={cb_metrics.get('failure_count')}")
    
    # Cleanup
    orchestrator.shutdown()


def demonstrate_chaos_engineering():
    """Demonstrate chaos engineering for testing"""
    logger.info("\n=== Chaos Engineering Demo ===")
    
    # Enable chaos engine (only for testing!)
    chaos_engine = ChaosEngine(enabled=True, safe_mode=False)
    
    # Configure chaos scenarios
    latency_chaos = ChaosConfig(
        strategy=ChaosStrategy.LATENCY,
        probability=0.3,
        latency_ms=500
    )
    
    error_chaos = ChaosConfig(
        strategy=ChaosStrategy.ERROR,
        probability=0.2,
        error_type=ConnectionError,
        error_message="Chaos: Simulated connection failure"
    )
    
    def normal_operation(value):
        return f"Processed: {value}"
    
    logger.info("\nRunning operations with chaos injection:")
    for i in range(10):
        try:
            # Randomly choose chaos type
            import random
            chaos_config = random.choice([latency_chaos, error_chaos, None])
            
            if chaos_config:
                result = chaos_engine.inject(
                    normal_operation,
                    f"Request {i}",
                    chaos_config=chaos_config
                )
            else:
                result = normal_operation(f"Request {i}")
            
            logger.info(f"  Request {i}: {result}")
            
        except Exception as e:
            logger.warning(f"  Request {i} failed: {e}")
    
    # Check chaos statistics
    stats = chaos_engine.get_statistics()
    logger.info(f"\nChaos Statistics:")
    logger.info(f"  Total calls: {stats['total_calls']}")
    logger.info(f"  Chaos injected: {stats['chaos_injected']}")
    logger.info(f"  Injection rate: {stats['injection_rate']:.2%}")
    logger.info(f"  By strategy: {stats['by_strategy']}")


async def demonstrate_async_resilience():
    """Demonstrate async resilience patterns"""
    logger.info("\n=== Async Resilience Demo ===")
    
    # Create async retry policy
    retry_policy = RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=0.2
    )
    
    call_count = 0
    
    async def async_flaky_service():
        nonlocal call_count
        call_count += 1
        logger.info(f"  Async attempt {call_count}")
        if call_count < 2:
            raise ConnectionError("Async service failed")
        return "Async success!"
    
    try:
        result = await retry_policy.execute_async(async_flaky_service)
        logger.info(f"  Result: {result}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
    
    # Async bulkhead
    bulkhead = SemaphoreBulkhead(
        name="async_bulkhead",
        max_concurrent=2
    )
    
    async def async_operation(op_id):
        logger.info(f"  Starting async operation {op_id}")
        await asyncio.sleep(0.3)
        logger.info(f"  Completed async operation {op_id}")
        return f"Result {op_id}"
    
    # Run concurrent operations
    tasks = [
        bulkhead.execute_async(async_operation, i)
        for i in range(4)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"  Task {i} failed: {result}")
        else:
            logger.info(f"  Task {i}: {result}")


def main():
    """Main demonstration function"""
    logger.info("=" * 60)
    logger.info("RESILIENT RAG SYSTEM DEMONSTRATION")
    logger.info("=" * 60)
    
    # Run demonstrations
    demonstrate_basic_resilience()
    demonstrate_resilient_rag()
    demonstrate_chaos_engineering()
    
    # Run async demonstrations
    logger.info("\n" + "=" * 60)
    asyncio.run(demonstrate_async_resilience())
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()