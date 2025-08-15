"""
Timeout Manager for External Calls

Provides configurable timeout management for all external operations.
"""

import time
import signal
import threading
import asyncio
import functools
from typing import Callable, Any, Optional, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    def __init__(self, message: str, elapsed_time: Optional[float] = None):
        super().__init__(message)
        self.elapsed_time = elapsed_time


class TimeoutManager:
    """
    Manages timeouts for external calls
    
    Provides multiple timeout strategies including signal-based,
    thread-based, and async timeout handling.
    """
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        use_signal: bool = False,
        log_timeouts: bool = True
    ):
        """
        Initialize timeout manager
        
        Args:
            default_timeout: Default timeout in seconds
            use_signal: Use signal-based timeout (Unix only, not thread-safe)
            log_timeouts: Whether to log timeout events
        """
        self.default_timeout = default_timeout
        self.use_signal = use_signal
        self.log_timeouts = log_timeouts
        self._timeout_stats = {
            'total_calls': 0,
            'timeout_count': 0,
            'total_execution_time': 0.0
        }
        self._stats_lock = threading.Lock()
    
    def execute(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with timeout
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutException: If function exceeds timeout
        """
        timeout = timeout or self.default_timeout
        
        if self.use_signal and threading.current_thread() is threading.main_thread():
            return self._execute_with_signal(func, args, kwargs, timeout)
        else:
            return self._execute_with_thread(func, args, kwargs, timeout)
    
    async def execute_async(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute async function with timeout
        
        Args:
            func: Async function to execute
            *args: Function arguments
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutException: If function exceeds timeout
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        self._record_call_start()
        
        try:
            # Use asyncio timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            self._record_call_success(execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            self._record_timeout()
            
            if self.log_timeouts:
                logger.warning(
                    f"Async operation timed out after {elapsed_time:.2f}s "
                    f"(timeout: {timeout}s)"
                )
            
            raise TimeoutException(
                f"Operation timed out after {elapsed_time:.2f} seconds",
                elapsed_time=elapsed_time
            )
    
    def _execute_with_signal(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float
    ) -> Any:
        """Execute with signal-based timeout (Unix only)"""
        
        def timeout_handler(signum, frame):
            raise TimeoutException(f"Operation timed out after {timeout} seconds")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        start_time = time.time()
        self._record_call_start()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._record_call_success(execution_time)
            return result
            
        except TimeoutException:
            elapsed_time = time.time() - start_time
            self._record_timeout()
            
            if self.log_timeouts:
                logger.warning(
                    f"Operation timed out after {elapsed_time:.2f}s "
                    f"(timeout: {timeout}s)"
                )
            
            raise TimeoutException(
                f"Operation timed out after {elapsed_time:.2f} seconds",
                elapsed_time=elapsed_time
            )
            
        finally:
            # Reset signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _execute_with_thread(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float
    ) -> Any:
        """Execute with thread-based timeout"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        
        start_time = time.time()
        self._record_call_start()
        
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            elapsed_time = time.time() - start_time
            self._record_timeout()
            
            if self.log_timeouts:
                logger.warning(
                    f"Operation timed out after {elapsed_time:.2f}s "
                    f"(timeout: {timeout}s)"
                )
            
            # Note: Thread continues running in background
            raise TimeoutException(
                f"Operation timed out after {elapsed_time:.2f} seconds",
                elapsed_time=elapsed_time
            )
        
        execution_time = time.time() - start_time
        self._record_call_success(execution_time)
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    @contextmanager
    def timeout_context(self, timeout: Optional[float] = None):
        """
        Context manager for timeout operations
        
        Usage:
            with timeout_manager.timeout_context(5.0):
                # Code that must complete within 5 seconds
                pass
        """
        timeout = timeout or self.default_timeout
        
        if self.use_signal and threading.current_thread() is threading.main_thread():
            def timeout_handler(signum, frame):
                raise TimeoutException(f"Operation timed out after {timeout} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            start_time = time.time()
            
            try:
                yield
            except TimeoutException:
                elapsed_time = time.time() - start_time
                if self.log_timeouts:
                    logger.warning(f"Context timed out after {elapsed_time:.2f}s")
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # For thread-based timeout, we can't interrupt the code
            # So we just yield and let the caller handle timeout
            yield
    
    def decorator(self, timeout: Optional[float] = None):
        """
        Decorator for adding timeout to functions
        
        Usage:
            @timeout_manager.decorator(timeout=5.0)
            def my_function():
                pass
        """
        def decorator_wrapper(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(func, *args, timeout=timeout, **kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute_async(func, *args, timeout=timeout, **kwargs)
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator_wrapper
    
    def _record_call_start(self):
        """Record start of call"""
        with self._stats_lock:
            self._timeout_stats['total_calls'] += 1
    
    def _record_call_success(self, execution_time: float):
        """Record successful call"""
        with self._stats_lock:
            self._timeout_stats['total_execution_time'] += execution_time
    
    def _record_timeout(self):
        """Record timeout"""
        with self._stats_lock:
            self._timeout_stats['timeout_count'] += 1
    
    def get_stats(self) -> dict:
        """Get timeout statistics"""
        with self._stats_lock:
            total_calls = self._timeout_stats['total_calls']
            timeout_count = self._timeout_stats['timeout_count']
            
            return {
                'total_calls': total_calls,
                'timeout_count': timeout_count,
                'timeout_rate': timeout_count / total_calls if total_calls > 0 else 0.0,
                'average_execution_time': (
                    self._timeout_stats['total_execution_time'] / 
                    (total_calls - timeout_count)
                    if (total_calls - timeout_count) > 0 else 0.0
                )
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self._stats_lock:
            self._timeout_stats = {
                'total_calls': 0,
                'timeout_count': 0,
                'total_execution_time': 0.0
            }


class AdaptiveTimeoutManager(TimeoutManager):
    """
    Adaptive timeout manager that adjusts timeouts based on historical performance
    """
    
    def __init__(
        self,
        initial_timeout: float = 30.0,
        min_timeout: float = 1.0,
        max_timeout: float = 300.0,
        adjustment_factor: float = 1.5,
        percentile: float = 0.95,
        window_size: int = 100
    ):
        """
        Initialize adaptive timeout manager
        
        Args:
            initial_timeout: Initial timeout value
            min_timeout: Minimum allowed timeout
            max_timeout: Maximum allowed timeout
            adjustment_factor: Factor for timeout adjustment
            percentile: Percentile of execution times to use for timeout
            window_size: Size of sliding window for metrics
        """
        super().__init__(default_timeout=initial_timeout)
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.adjustment_factor = adjustment_factor
        self.percentile = percentile
        self.window_size = window_size
        
        self._execution_times = []
        self._adaptive_timeout = initial_timeout
    
    def _record_call_success(self, execution_time: float):
        """Record successful call and adjust timeout"""
        super()._record_call_success(execution_time)
        
        with self._stats_lock:
            self._execution_times.append(execution_time)
            
            # Maintain window size
            if len(self._execution_times) > self.window_size:
                self._execution_times.pop(0)
            
            # Adjust timeout based on percentile
            if len(self._execution_times) >= 10:  # Need minimum samples
                sorted_times = sorted(self._execution_times)
                percentile_index = int(len(sorted_times) * self.percentile)
                percentile_time = sorted_times[min(percentile_index, len(sorted_times) - 1)]
                
                # Apply adjustment factor
                new_timeout = percentile_time * self.adjustment_factor
                
                # Apply bounds
                self._adaptive_timeout = max(
                    self.min_timeout,
                    min(self.max_timeout, new_timeout)
                )
                
                self.default_timeout = self._adaptive_timeout
    
    def get_adaptive_timeout(self) -> float:
        """Get current adaptive timeout value"""
        return self._adaptive_timeout