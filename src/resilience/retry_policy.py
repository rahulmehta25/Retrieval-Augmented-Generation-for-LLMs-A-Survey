"""
Retry Policy Implementation with Exponential Backoff and Jitter

Provides configurable retry mechanisms for handling transient failures.
"""

import time
import random
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional, List, Type, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


@dataclass
class RetryContext:
    """Context information for retry attempts"""
    attempt_number: int
    total_attempts: int
    elapsed_time: float
    last_exception: Optional[Exception]
    retry_delay: float
    strategy: RetryStrategy


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter
    
    Implements various retry strategies with configurable parameters
    for handling transient failures in distributed systems.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retriable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retriable_exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[RetryContext], None]] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize retry policy
        
        Args:
            max_attempts: Maximum number of retry attempts
            strategy: Retry strategy to use
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter to delays
            jitter_factor: Jitter factor (0.0 to 1.0)
            retriable_exceptions: Exceptions that trigger retry
            non_retriable_exceptions: Exceptions that should not be retried
            on_retry: Callback function called on each retry
            timeout: Overall timeout for all retries
        """
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = min(max(jitter_factor, 0.0), 1.0)
        self.retriable_exceptions = retriable_exceptions or [Exception]
        self.non_retriable_exceptions = non_retriable_exceptions or []
        self.on_retry = on_retry
        self.timeout = timeout
        
        # Fibonacci sequence cache
        self._fibonacci_cache = [1, 1]
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry policy
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Check overall timeout
                if self.timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= self.timeout:
                        raise TimeoutError(f"Retry timeout exceeded: {elapsed:.2f}s")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Success - log if this was a retry
                if attempt > 1:
                    logger.info(f"Retry successful on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception should be retried
                if not self._should_retry(e):
                    logger.error(f"Non-retriable exception: {e}")
                    raise
                
                # Check if we have more attempts
                if attempt >= self.max_attempts:
                    logger.error(f"Max retry attempts ({self.max_attempts}) exceeded")
                    raise
                
                # Calculate retry delay
                delay = self._calculate_delay(attempt)
                
                # Create retry context
                context = RetryContext(
                    attempt_number=attempt,
                    total_attempts=self.max_attempts,
                    elapsed_time=time.time() - start_time,
                    last_exception=e,
                    retry_delay=delay,
                    strategy=self.strategy
                )
                
                # Call retry callback
                if self.on_retry:
                    try:
                        self.on_retry(context)
                    except Exception as callback_error:
                        logger.error(f"Error in retry callback: {callback_error}")
                
                # Log retry attempt
                logger.warning(
                    f"Retry attempt {attempt}/{self.max_attempts} after {delay:.2f}s delay. "
                    f"Error: {e}"
                )
                
                # Wait before retry
                time.sleep(delay)
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with retry policy
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Check overall timeout
                if self.timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= self.timeout:
                        raise TimeoutError(f"Retry timeout exceeded: {elapsed:.2f}s")
                
                # Execute async function
                result = await func(*args, **kwargs)
                
                # Success - log if this was a retry
                if attempt > 1:
                    logger.info(f"Retry successful on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception should be retried
                if not self._should_retry(e):
                    logger.error(f"Non-retriable exception: {e}")
                    raise
                
                # Check if we have more attempts
                if attempt >= self.max_attempts:
                    logger.error(f"Max retry attempts ({self.max_attempts}) exceeded")
                    raise
                
                # Calculate retry delay
                delay = self._calculate_delay(attempt)
                
                # Create retry context
                context = RetryContext(
                    attempt_number=attempt,
                    total_attempts=self.max_attempts,
                    elapsed_time=time.time() - start_time,
                    last_exception=e,
                    retry_delay=delay,
                    strategy=self.strategy
                )
                
                # Call retry callback
                if self.on_retry:
                    try:
                        if asyncio.iscoroutinefunction(self.on_retry):
                            await self.on_retry(context)
                        else:
                            self.on_retry(context)
                    except Exception as callback_error:
                        logger.error(f"Error in retry callback: {callback_error}")
                
                # Log retry attempt
                logger.warning(
                    f"Retry attempt {attempt}/{self.max_attempts} after {delay:.2f}s delay. "
                    f"Error: {e}"
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry"""
        # Check non-retriable exceptions first
        for non_retriable_type in self.non_retriable_exceptions:
            if isinstance(exception, non_retriable_type):
                return False
        
        # Check retriable exceptions
        for retriable_type in self.retriable_exceptions:
            if isinstance(exception, retriable_type):
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.initial_delay
        
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay * attempt
        
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.initial_delay * self._get_fibonacci(attempt)
        
        else:  # CUSTOM or unknown
            delay = self.initial_delay
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter if enabled
        if self.jitter:
            jitter_amount = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached)"""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n]
    
    @classmethod
    def with_exponential_backoff(
        cls,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ) -> 'RetryPolicy':
        """Create retry policy with exponential backoff"""
        return cls(
            max_attempts=max_attempts,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter
        )
    
    @classmethod
    def with_fixed_delay(
        cls,
        max_attempts: int = 3,
        delay: float = 1.0
    ) -> 'RetryPolicy':
        """Create retry policy with fixed delay"""
        return cls(
            max_attempts=max_attempts,
            strategy=RetryStrategy.FIXED_DELAY,
            initial_delay=delay,
            jitter=False
        )
    
    def with_circuit_breaker(self, circuit_breaker: 'CircuitBreaker') -> 'RetryPolicy':
        """
        Combine retry policy with circuit breaker
        
        Args:
            circuit_breaker: Circuit breaker to use
            
        Returns:
            New retry policy that respects circuit breaker state
        """
        original_execute = self.execute
        
        def execute_with_breaker(func: Callable, *args, **kwargs):
            def wrapped():
                return original_execute(func, *args, **kwargs)
            return circuit_breaker.call(wrapped)
        
        # Create new instance with modified execute
        new_policy = RetryPolicy(
            max_attempts=self.max_attempts,
            strategy=self.strategy,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            jitter_factor=self.jitter_factor,
            retriable_exceptions=self.retriable_exceptions,
            non_retriable_exceptions=self.non_retriable_exceptions,
            on_retry=self.on_retry,
            timeout=self.timeout
        )
        new_policy.execute = execute_with_breaker
        
        return new_policy