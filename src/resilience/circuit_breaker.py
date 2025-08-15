"""
Circuit Breaker Pattern Implementation

Provides protection against cascading failures by monitoring call failures
and temporarily blocking calls when a threshold is exceeded.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Calls blocked due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit Breaker implementation with half-open state support
    
    The circuit breaker monitors calls and their failures:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are blocked
    - HALF_OPEN: Testing recovery with limited calls
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2,
        half_open_max_calls: int = 3,
        failure_rate_threshold: float = 0.5,
        minimum_number_of_calls: int = 10,
        sliding_window_size: int = 100
    ):
        """
        Initialize circuit breaker
        
        Args:
            name: Circuit breaker name for identification
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close from half-open
            half_open_max_calls: Max concurrent calls in half-open state
            failure_rate_threshold: Failure rate to trigger open state
            minimum_number_of_calls: Minimum calls before evaluating failure rate
            sliding_window_size: Size of sliding window for metrics
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.failure_rate_threshold = failure_rate_threshold
        self.minimum_number_of_calls = minimum_number_of_calls
        self.sliding_window_size = sliding_window_size
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
        
        # Sliding window for metrics
        self._call_results = []  # List of (timestamp, success: bool)
        self._state_change_listeners = []
        
    @property
    def state(self) -> CircuitBreakerState:
        """Get current state, checking for automatic transitions"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
            return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: If function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    retry_after=retry_after
                )
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached",
                        retry_after=1.0
                    )
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: If function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    retry_after=retry_after
                )
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached",
                        retry_after=1.0
                    )
                self._half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self._record_call(True)
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self._record_call(False)
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count += 1
                if self._should_open():
                    self._transition_to_open()
    
    def _should_open(self) -> bool:
        """Check if circuit should open based on failures"""
        # Check absolute failure threshold
        if self._failure_count >= self.failure_threshold:
            return True
        
        # Check failure rate in sliding window
        if len(self._call_results) >= self.minimum_number_of_calls:
            failure_rate = self._calculate_failure_rate()
            if failure_rate >= self.failure_rate_threshold:
                return True
        
        return False
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in sliding window"""
        if not self._call_results:
            return 0.0
        
        failures = sum(1 for _, success in self._call_results if not success)
        return failures / len(self._call_results)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _get_retry_after(self) -> Optional[float]:
        """Calculate time until next retry attempt"""
        if self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            remaining = self.recovery_timeout - elapsed
            return max(0, remaining)
        return None
    
    def _record_call(self, success: bool):
        """Record call result in sliding window"""
        timestamp = time.time()
        self._call_results.append((timestamp, success))
        
        # Maintain sliding window size
        if len(self._call_results) > self.sliding_window_size:
            self._call_results.pop(0)
        
        # Remove old entries outside time window
        cutoff_time = timestamp - (self.recovery_timeout * 2)
        self._call_results = [
            (ts, result) for ts, result in self._call_results
            if ts > cutoff_time
        ]
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN")
        self._state = CircuitBreakerState.OPEN
        self._notify_state_change(CircuitBreakerState.OPEN)
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._notify_state_change(CircuitBreakerState.CLOSED)
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        self._state = CircuitBreakerState.HALF_OPEN
        self._success_count = 0
        self._half_open_calls = 0
        self._notify_state_change(CircuitBreakerState.HALF_OPEN)
    
    def _notify_state_change(self, new_state: CircuitBreakerState):
        """Notify listeners of state change"""
        for listener in self._state_change_listeners:
            try:
                listener(self.name, new_state)
            except Exception as e:
                logger.error(f"Error notifying state change listener: {e}")
    
    def add_state_change_listener(self, listener: Callable):
        """Add listener for state changes"""
        self._state_change_listeners.append(listener)
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            self._transition_to_closed()
            self._call_results.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'failure_rate': self._calculate_failure_rate(),
                'total_calls': len(self._call_results),
                'last_failure_time': self._last_failure_time,
                'half_open_calls': self._half_open_calls
            }