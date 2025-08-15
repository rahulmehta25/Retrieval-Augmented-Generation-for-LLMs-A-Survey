"""
Resilience Decorators for Easy Integration

Provides decorators to add resilience patterns to functions.
"""

import functools
import asyncio
from typing import Optional, Callable, Any, Type, List
from .circuit_breaker import CircuitBreaker
from .retry_policy import RetryPolicy, RetryStrategy
from .bulkhead import SemaphoreBulkhead, ThreadPoolBulkhead
from .timeout_manager import TimeoutManager
from .fallback import FallbackHandler, FallbackStrategy


def resilient(
    # Circuit breaker settings
    circuit_breaker: Optional[CircuitBreaker] = None,
    circuit_breaker_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    
    # Retry settings
    retry_policy: Optional[RetryPolicy] = None,
    max_attempts: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    
    # Bulkhead settings
    bulkhead: Optional[SemaphoreBulkhead] = None,
    bulkhead_name: Optional[str] = None,
    max_concurrent: int = 10,
    
    # Timeout settings
    timeout: Optional[float] = None,
    
    # Fallback settings
    fallback_handler: Optional[FallbackHandler] = None,
    fallback_value: Any = None,
    fallback_function: Optional[Callable] = None,
    
    # Exception handling
    expected_exceptions: List[Type[Exception]] = None,
    
    # Monitoring
    enable_monitoring: bool = True
):
    """
    Decorator that adds multiple resilience patterns to a function
    
    Usage:
        @resilient(
            max_attempts=3,
            timeout=5.0,
            fallback_value="default"
        )
        def my_function():
            pass
    
    Args:
        circuit_breaker: Existing circuit breaker instance
        circuit_breaker_name: Name for new circuit breaker
        failure_threshold: Circuit breaker failure threshold
        recovery_timeout: Circuit breaker recovery timeout
        retry_policy: Existing retry policy instance
        max_attempts: Maximum retry attempts
        retry_strategy: Retry strategy
        initial_delay: Initial retry delay
        max_delay: Maximum retry delay
        bulkhead: Existing bulkhead instance
        bulkhead_name: Name for new bulkhead
        max_concurrent: Maximum concurrent calls
        timeout: Execution timeout
        fallback_handler: Existing fallback handler
        fallback_value: Static fallback value
        fallback_function: Fallback function
        expected_exceptions: Exceptions to handle
        enable_monitoring: Enable metrics collection
    """
    
    def decorator(func: Callable) -> Callable:
        # Create or use existing components
        _circuit_breaker = circuit_breaker
        if not _circuit_breaker and circuit_breaker_name:
            _circuit_breaker = CircuitBreaker(
                name=circuit_breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=tuple(expected_exceptions) if expected_exceptions else Exception
            )
        
        _retry_policy = retry_policy
        if not _retry_policy:
            _retry_policy = RetryPolicy(
                max_attempts=max_attempts,
                strategy=retry_strategy,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retriable_exceptions=expected_exceptions
            )
        
        _bulkhead = bulkhead
        if not _bulkhead and bulkhead_name:
            _bulkhead = SemaphoreBulkhead(
                name=bulkhead_name,
                max_concurrent=max_concurrent
            )
        
        _timeout_manager = TimeoutManager(default_timeout=timeout) if timeout else None
        
        _fallback_handler = fallback_handler
        if not _fallback_handler and (fallback_value is not None or fallback_function):
            _fallback_handler = FallbackHandler(
                strategy=FallbackStrategy.STATIC_VALUE if fallback_value is not None else FallbackStrategy.CUSTOM_HANDLER,
                fallback_value=fallback_value,
                fallback_function=fallback_function
            )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def execute():
                # Apply timeout if configured
                if _timeout_manager:
                    return _timeout_manager.execute(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            # Apply bulkhead if configured
            if _bulkhead:
                execute_with_bulkhead = lambda: _bulkhead.execute(execute)
            else:
                execute_with_bulkhead = execute
            
            # Apply retry if configured
            if _retry_policy:
                execute_with_retry = lambda: _retry_policy.execute(execute_with_bulkhead)
            else:
                execute_with_retry = execute_with_bulkhead
            
            # Apply circuit breaker if configured
            if _circuit_breaker:
                execute_with_breaker = lambda: _circuit_breaker.call(execute_with_retry)
            else:
                execute_with_breaker = execute_with_retry
            
            # Apply fallback if configured
            if _fallback_handler:
                try:
                    return execute_with_breaker()
                except Exception as e:
                    if expected_exceptions and not any(isinstance(e, exc) for exc in expected_exceptions):
                        raise
                    return _fallback_handler.execute(
                        lambda: None,  # Primary already failed
                        cache_key=f"{func.__name__}_{args}_{kwargs}"
                    )
            else:
                return execute_with_breaker()
        
        return wrapper
    
    return decorator


def resilient_async(
    # Circuit breaker settings
    circuit_breaker: Optional[CircuitBreaker] = None,
    circuit_breaker_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    
    # Retry settings
    retry_policy: Optional[RetryPolicy] = None,
    max_attempts: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    
    # Bulkhead settings
    bulkhead: Optional[SemaphoreBulkhead] = None,
    bulkhead_name: Optional[str] = None,
    max_concurrent: int = 10,
    
    # Timeout settings
    timeout: Optional[float] = None,
    
    # Fallback settings
    fallback_handler: Optional[FallbackHandler] = None,
    fallback_value: Any = None,
    fallback_function: Optional[Callable] = None,
    
    # Exception handling
    expected_exceptions: List[Type[Exception]] = None,
    
    # Monitoring
    enable_monitoring: bool = True
):
    """
    Async version of resilient decorator
    
    Usage:
        @resilient_async(
            max_attempts=3,
            timeout=5.0,
            fallback_value="default"
        )
        async def my_async_function():
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        # Create or use existing components
        _circuit_breaker = circuit_breaker
        if not _circuit_breaker and circuit_breaker_name:
            _circuit_breaker = CircuitBreaker(
                name=circuit_breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=tuple(expected_exceptions) if expected_exceptions else Exception
            )
        
        _retry_policy = retry_policy
        if not _retry_policy:
            _retry_policy = RetryPolicy(
                max_attempts=max_attempts,
                strategy=retry_strategy,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retriable_exceptions=expected_exceptions
            )
        
        _bulkhead = bulkhead
        if not _bulkhead and bulkhead_name:
            _bulkhead = SemaphoreBulkhead(
                name=bulkhead_name,
                max_concurrent=max_concurrent
            )
        
        _timeout_manager = TimeoutManager(default_timeout=timeout) if timeout else None
        
        _fallback_handler = fallback_handler
        if not _fallback_handler and (fallback_value is not None or fallback_function):
            _fallback_handler = FallbackHandler(
                strategy=FallbackStrategy.STATIC_VALUE if fallback_value is not None else FallbackStrategy.CUSTOM_HANDLER,
                fallback_value=fallback_value,
                fallback_function=fallback_function
            )
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def execute():
                # Apply timeout if configured
                if _timeout_manager:
                    return await _timeout_manager.execute_async(func, *args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            
            # Apply bulkhead if configured
            if _bulkhead:
                execute_with_bulkhead = lambda: _bulkhead.execute_async(execute)
            else:
                execute_with_bulkhead = execute
            
            # Apply retry if configured
            if _retry_policy:
                execute_with_retry = lambda: _retry_policy.execute_async(execute_with_bulkhead)
            else:
                execute_with_retry = execute_with_bulkhead
            
            # Apply circuit breaker if configured
            if _circuit_breaker:
                execute_with_breaker = lambda: _circuit_breaker.call_async(execute_with_retry)
            else:
                execute_with_breaker = execute_with_retry
            
            # Apply fallback if configured
            if _fallback_handler:
                try:
                    return await execute_with_breaker()
                except Exception as e:
                    if expected_exceptions and not any(isinstance(e, exc) for exc in expected_exceptions):
                        raise
                    return await _fallback_handler.execute_async(
                        lambda: None,  # Primary already failed
                        cache_key=f"{func.__name__}_{args}_{kwargs}"
                    )
            else:
                return await execute_with_breaker()
        
        return wrapper
    
    return decorator


# Convenience decorators for specific patterns

def with_retry(max_attempts: int = 3, **kwargs):
    """Decorator for retry only"""
    return resilient(max_attempts=max_attempts, **kwargs)


def with_circuit_breaker(name: str, **kwargs):
    """Decorator for circuit breaker only"""
    return resilient(circuit_breaker_name=name, **kwargs)


def with_timeout(timeout: float, **kwargs):
    """Decorator for timeout only"""
    return resilient(timeout=timeout, **kwargs)


def with_fallback(fallback_value: Any = None, fallback_function: Callable = None, **kwargs):
    """Decorator for fallback only"""
    return resilient(fallback_value=fallback_value, fallback_function=fallback_function, **kwargs)


def with_bulkhead(name: str, max_concurrent: int = 10, **kwargs):
    """Decorator for bulkhead only"""
    return resilient(bulkhead_name=name, max_concurrent=max_concurrent, **kwargs)