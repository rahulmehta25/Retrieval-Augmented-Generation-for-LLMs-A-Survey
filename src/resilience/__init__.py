"""
Resilience Module for RAG System

This module provides fault tolerance and resilience patterns including:
- Circuit breakers with half-open state support
- Retry policies with exponential backoff and jitter
- Bulkhead pattern for resource isolation
- Timeout management
- Fallback mechanisms
- Health checking and recovery
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerOpenException
from .retry_policy import RetryPolicy, RetryContext, RetryStrategy
from .bulkhead import Bulkhead, BulkheadFullException, ThreadPoolBulkhead, SemaphoreBulkhead
from .timeout_manager import TimeoutManager, TimeoutException
from .fallback import FallbackHandler, FallbackStrategy
from .health_checker import HealthChecker, HealthStatus, HealthCheck
from .resilience_decorator import resilient, resilient_async
from .monitoring import ResilienceMonitor, MetricsCollector
from .chaos import ChaosEngine, ChaosStrategy
from .tracing import DistributedTracer, TraceContext

__all__ = [
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerOpenException',
    
    # Retry Policy
    'RetryPolicy',
    'RetryContext',
    'RetryStrategy',
    
    # Bulkhead
    'Bulkhead',
    'BulkheadFullException',
    'ThreadPoolBulkhead',
    'SemaphoreBulkhead',
    
    # Timeout
    'TimeoutManager',
    'TimeoutException',
    
    # Fallback
    'FallbackHandler',
    'FallbackStrategy',
    
    # Health Check
    'HealthChecker',
    'HealthStatus',
    'HealthCheck',
    
    # Decorators
    'resilient',
    'resilient_async',
    
    # Monitoring
    'ResilienceMonitor',
    'MetricsCollector',
    
    # Chaos Engineering
    'ChaosEngine',
    'ChaosStrategy',
    
    # Distributed Tracing
    'DistributedTracer',
    'TraceContext',
]